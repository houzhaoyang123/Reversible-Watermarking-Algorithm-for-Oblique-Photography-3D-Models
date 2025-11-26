import open3d as o3d
import sys
import numpy as np
from PIL import Image  # 库的原因读成True/False  用于处理图片
import math
import cv2

def Get_OBJ(ori_obj, ori_picture):
    obj_mesh = o3d.io.read_triangle_mesh(ori_obj)    #加载三角网格
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
    point = np.asarray(cloud_in.points)   #将点数据转化成数组格式
    point_size = point.shape[0]
    # 计算法向量，K近邻搜索近邻点
    cloud_in.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn_num))   #estimate_normals函数用于估计点云中每个点的法线。它接受一个search_param参数，用于领域搜索的参数
    # 计算邻域内法向量夹角的均值
    normal = np.asarray(cloud_in.normals)  # 从pcd中提取法线信息
    normal_angle = np.zeros(point_size)  # 用来存储邻域法向量夹角均值的容器
    tree = o3d.geometry.KDTreeFlann(cloud_in)  # 建立KD树索引   目的是遍历所有的点

    for i in range(point_size):
        [_, idx, _] = tree.search_knn_vector_3d(point[i], knn_num + 1)    #返回找到的每个点周围的36个点
        current_normal = normal[i]
        temp = []
        for j in range(1, len(idx)):
            knn_normal = normal[idx[j]]
            temp.append(get_angle_vector(current_normal, knn_normal))
        normal_angle[i] = np.mean(np.array(temp, dtype=object))
        temp.clear()
    keypoint_idx = np.where(normal_angle >= angle_thre)[0]  # 满足指定夹角阈值，位于特征明显区域的点的索引,特殊点的索引
    keypoint = pcd.select_by_index(keypoint_idx)    #会根据索引选择一个新的点云数据
    o3d.io.write_point_cloud("selected_points.pcd",keypoint)
    return keypoint_idx,keypoint

'''嵌入水印'''
def get_newR(r_list,t,wm_value):
    new_R = []
    for i in range(len(r_list)):
        value = (int(r_list[i] * math.pow(10, t)) + (wm_value[int(r_list[i]*10) % len(wm_value)]
                + r_list[i]*math.pow(10,t)-int(r_list[i]*math.pow(10,t)))/10) / math.pow(10, t)
        new_R.append(value)
    return new_R

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
    o3d.io.write_triangle_mesh("newModel001.ply", pcd1)

if __name__ == '__main__':

    pcd = o3d.io.read_point_cloud("model1220.pcd")
    keypoint_idx, keypoint_cloud = keypoint_extract_by_angle(pcd, knn_num=35, angle_thre=88)

    wm = cv2.imread(r"wm32-32.bmp", 0)  # 读取原始水印图像
    wm[wm > 0] = 1
    wm = wm.reshape(-1)

    feature, d, obj_sjm, uv, wenli_iamge = Get_OBJ("Model.obj", "Model_0.jpg")
    dindex = []
    for j in range(len(keypoint_idx)):
        ds1 = np.sqrt(np.sum(np.square(d[keypoint_idx[j]] - d), axis=1))   #square是各元素的平方
        ds1[keypoint_idx] = -99
        ds1[np.isnan(ds1)] = -99
        index = np.argsort(ds1)      #元素值从小到大的排列后的索引

        dindex.append([keypoint_idx[j]] + list(index[- 2:]))
        d[keypoint_idx[j]] = np.NaN
        d[index[- 2:]] = np.NaN
        if j % 1000 == 0:
            print(j)

    arr_index = np.asarray(dindex)
    arr_indexColumn = arr_index[:,0]
    arr_indexColumn1_3 = arr_index[:, 1:3]
    modify_xyzIdx = list(arr_indexColumn1_3.flatten())  # 列表化，做索引

    feature, d1, obj_sjm, uv, wenli_iamge = Get_OBJ("Model.obj", "Model_0.jpg")
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

    '''修改R小数位'''
    t1,t2 = 2,7
    newFirstR = get_newR(arr_radius,t1,wm)
    newSecondR =get_newR(newFirstR,t2,wm)
    arr_newR = np.asarray(newSecondR)

    '''得到新的xyz坐标'''
    new_x = arr_newR * np.sin(arr_cphs) * np.cos(arr_thetas)
    new_y = arr_newR * np.sin(arr_cphs) * np.sin(arr_thetas)
    new_z = arr_newR * np.cos(arr_cphs)

    ori_newxyz = np.array([new_x,new_y,new_z])
    ori_newxyzTran = np.transpose(ori_newxyz)  #翻转xyz的组合

    arr_indexColumnX2 = arr_indexColumn.repeat(2)   #扩列至2倍
    modify_xyz = d1[arr_indexColumnX2] + ori_newxyzTran

    '''待修改'''
    d1[modify_xyzIdx] = modify_xyz  #索引对应变量
    newModel = create_newobj(d1,"Model_0.jpg")

    print("ok")



