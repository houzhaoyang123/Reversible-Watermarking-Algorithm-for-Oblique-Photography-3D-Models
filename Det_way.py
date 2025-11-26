import open3d as o3d
import sys
import numpy as np
from PIL import Image  # 库的原因读成True/False
import math
import cv2
from collections import Counter

def Get_OBJ(ori_obj, ori_picture):
    obj_mesh = o3d.io.read_triangle_mesh(ori_obj)
    obj_picture01 = Image.open(ori_picture)
    if obj_mesh is None or obj_picture01 is None:
        sys.exit('Could not open {0}.'.format(ori_obj))
    obj_pc = np.asarray(obj_mesh.vertices)  # 顶点数据数组Vector3[]
    obj_sjm = np.asarray(obj_mesh.triangles)  # 三角形顶点索引数组，int[],可能是三角网点的顺序
    uv = np.asarray(obj_mesh.triangle_uvs)  # （uv）纹理坐标数组，Vector2[]
    wenli_iamge = obj_mesh.textures
    return obj_mesh, obj_pc, obj_sjm, uv, wenli_iamge

'''寻找特征点'''
def get_angle_vector(a, b, degrees=True):
    """计算法向量的夹角，
    :param a: 输入的法向量
    :param b: 输入的法向量
    :param degrees:
    :return: True输出为角度制，False输出为弧度制。
    """
    cos_a = np.inner(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    if cos_a < -1:
        cos_a = np.array([-1])
    if cos_a > 1:
        cos_a = np.array([1])
    rad = np.arccos(cos_a)  # 计算结果为弧度制
    deg = np.rad2deg(rad)  # 转化为角度制
    angle = deg if degrees else rad
    return angle

def keypoint_extract_by_angle(cloud_in, knn_num, angle_thre):
    """
     计算邻域法向量夹角均值，
    :param cloud_in: 输入点云
    :param knn_num: 邻域点个数
    :param angle_thre: 夹角均值阈值
    :return:
    """
    point = np.asarray(cloud_in.points)
    point_size = point.shape[0]
    # 计算法向量，K近邻搜索近邻点
    cloud_in.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn_num))
    # 计算邻域内法向量夹角的均值
    normal = np.asarray(cloud_in.normals)  # 从pcd中提取法线信息
    normal_angle = np.zeros(point_size)  # 用来存储邻域法向量夹角均值的容器
    tree = o3d.geometry.KDTreeFlann(cloud_in)  # 建立KD树索引

    for i in range(point_size):
        [_, idx, _] = tree.search_knn_vector_3d(point[i], knn_num + 1)
        current_normal = normal[i]
        temp = []
        for j in range(1, len(idx)):
            knn_normal = normal[idx[j]]
            temp.append(get_angle_vector(current_normal, knn_normal))
        normal_angle[i] = np.mean(np.array(temp, dtype=object))
        temp.clear()
    keypoint_idx = np.where(normal_angle >= angle_thre)[0]  # 满足指定夹角阈值，位于特征明显区域的点的索引
    keypoint = pcd.select_by_index(keypoint_idx)
    return keypoint_idx,keypoint

'''生成新的模型数据'''
def create_newobj(new_arr, ima_path):
    pcd1 = o3d.geometry.TriangleMesh()
    pcd1.vertices = o3d.utility.Vector3dVector(new_arr)
    pcd1.triangles = o3d.utility.Vector3iVector(obj_sjm)
    pcd1.triangle_uvs = o3d.open3d.utility.Vector2dVector(uv)
    tex_img = cv2.imread(ima_path)
    tex_img = cv2.cvtColor(tex_img, cv2.COLOR_BGR2RGB)
    tex_img = cv2.flip(tex_img, 0)
    pcd1.textures = [o3d.geometry.Image(tex_img)]
    pcd1.compute_vertex_normals()
    o3d.io.write_triangle_mesh("fuyuan_newModel.obj", pcd1)

if __name__ == '__main__':
    pcd = o3d.io.read_point_cloud("model1220_09.pcd")
    keypoint_idx, keypoint_cloud = keypoint_extract_by_angle(pcd, knn_num=35, angle_thre=88)

    feature, d, obj_sjm, uv, wenli_iamge = Get_OBJ("newModel001.ply", "Model_0.jpg")

    dindex = []
    for j in range(len(keypoint_idx)):
        ds1 = np.sqrt(np.sum(np.square(d[keypoint_idx[j]] - d), axis=1))
        ds1[keypoint_idx] = -99
        ds1[np.isnan(ds1)] = -99
        index = np.argsort(ds1)

        dindex.append([keypoint_idx[j]] + list(index[- 2:]))
        d[keypoint_idx[j]] = np.NaN
        d[index[- 2:]] = np.NaN
        if j % 1000 == 0:
            print(j)
    arr_index = np.asarray(dindex)
    arr_indexColumn = arr_index[:, 0]
    arr_indexColumn1_3 = arr_index[:,1:3]
    modify_xyzIdx = list(arr_indexColumn1_3.flatten())  #后两列 索引

    feature, d1, obj_sjm, uv, wenli_iamge = Get_OBJ("newModel001.ply", "Model_0.jpg")
    ori_radius,thetas,cphs = [],[],[]
    for i in range(arr_index.shape[0]):
        for j in range(1, 3):
            r = math.sqrt((d1[:, 0][arr_index[i][j]] - d1[:, 0][arr_index[i][0]]) ** 2
                          + (d1[:, 1][arr_index[i][j]] - d1[:, 1][arr_index[i][0]]) ** 2
                          + (d1[:, 2][arr_index[i][j]] - d1[:, 2][arr_index[i][0]]) ** 2)

            theta = np.arctan((d1[:, 1][arr_index[i][j]] - d1[:, 1][arr_index[i][0]])
                              / (d1[:, 0][arr_index[i][j]] - d1[:, 0][arr_index[i][0]]))
            if d1[:, 0][arr_index[i][j]] - d1[:, 0][arr_index[i][0]] > 0:
                    theta = theta
            elif d1[:, 0][arr_index[i][j]] - d1[:, 0][arr_index[i][0]] < 0:
                    theta += math.pi

            cph = np.arccos((d1[:, 2][arr_index[i][j]] - d1[:, 2][arr_index[i][0]]) / r)

            thetas.append(theta)
            ori_radius.append(r)
            cphs.append(cph)
    arr_thetas = np.asarray(thetas)
    arr_radius = np.asarray(ori_radius)
    arr_cphs = np.asarray(cphs)

    '''提取水印'''
    t = 2
    wm1 = np.trunc(((arr_radius * math.pow(10, t)) - np.trunc(arr_radius * math.pow(10,t))) * 10)

    w = np.zeros(1024, dtype=int)
    wm_index = np.array(np.trunc(arr_radius * 10) % 1024, dtype=int)
    w = np.uint8(w)
    w[wm_index] = wm1
    w[w > 0] = 255
    w[w <= 0] = 0
    w = w.reshape(32,32)
    cv2.imwrite('newwm001_09.bmp', w)

    '''恢复半径'''
    t1,t2 = 2,6
    ori_firstR = (np.trunc(arr_radius * math.pow(10, t1)) + (arr_radius * math.pow(10, t1 + 1) - np.trunc(arr_radius * math.pow(10, t1 + 1)))) / math.pow(10, t1)
    ori_secondR = (np.trunc(ori_firstR * math.pow(10, t2)) +
    (ori_firstR * math.pow(10, t2 + 1) - np.trunc(ori_firstR * math.pow(10, t2 + 1)))) / math.pow(10, t2)
    # print("恢复半径是：",ori_secondR[97])

    '''选择不同的R值来恢复不同精度的原始数据'''
    arr_newR = ori_secondR
    new_x = arr_newR * np.sin(arr_cphs) * np.cos(arr_thetas)
    new_y = arr_newR * np.sin(arr_cphs) * np.sin(arr_thetas)
    new_z = arr_newR * np.cos(arr_cphs)

    ori_newxyz = np.array([new_x,new_y,new_z])
    ori_newxyzTran = np.transpose(ori_newxyz)  #翻转xyz的组合

    arr_indexColumnX2 = arr_indexColumn.repeat(2)   #扩列至3倍
    modify_xyz = d1[arr_indexColumnX2] + ori_newxyzTran

    '''恢复原始模型'''
    d1[modify_xyzIdx] = modify_xyz  #可根据不同的t值恢复不同的数据

    newModel = create_newobj(d1,"Model_0.jpg")

    print('ok')



