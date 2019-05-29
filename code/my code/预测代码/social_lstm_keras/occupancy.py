
"""
Created on Mon Jul 31 14:23:47 2017
@author: Hao Xue
"""

import numpy as np
import math

# 矩形邻域
def get_rectangular_occupancy_map(frame_ID, ped_ID, dimensions, neighborhood_size, grid_size, data):
    """
    该函数用来计算每一帧中每个行人的矩形占用图
	
 
    参数:
        frame_ID: 帧id.
        ped_ID: 每个行人id
        dimensions : 图片帧的尺寸
        neighborhood_size : 标量值代表考虑的邻域大小，只考虑该范围内行人的影响
        grid_size : 标量值表示网格离散化的大小
        data: 数据格式，x，y坐标 
    """
  

    o_map = np.zeros((int(neighborhood_size / grid_size), int(neighborhood_size / grid_size)))
 

    ped_list = []

    # 搜索同一帧中的所有的行人
    for i in range(len(data[0])):
        if data[0][i] == frame_ID:
            ped_list.append(data[:, i])
  
    ped_list = np.reshape(ped_list, [-1, 4])
    #根据目标人与其他人的二维平面距离来判断，即若其他人跟目标的平面坐标距离<阈值，则认为其他人在目标邻域内，否则，不在邻域内
    if len(ped_list) == 0:
        print('no pedestrian in this frame!')
    elif len(ped_list) == 1:
        print('only one pedestrian in this frame!')
        return o_map
    else:
        for pedIndex in range(len(ped_list)):
            if ped_list[pedIndex][1] == ped_ID:
                current_x, current_y = ped_list[pedIndex][-1], ped_list[pedIndex][-2]
                width_low, width_high = current_x - neighborhood_size / 2, current_x + neighborhood_size / 2
                height_low, height_high = current_y - neighborhood_size / 2, current_y + neighborhood_size / 2
                current_index = pedIndex
        for otherIndex in range(len(ped_list)):
            if otherIndex != current_index:
                other_x, other_y = ped_list[otherIndex][-1], ped_list[otherIndex][-2]
                if other_x >= width_high or other_x < width_low or other_y >= height_high or other_y < height_low:
                    continue
                cell_x = int(np.floor((other_x - width_low) / grid_size))
                cell_y = int(np.floor((other_y - height_low) / grid_size))

                o_map[cell_x, cell_y] += 1
        #                o_map[cell_x + cell_y*grid_size] = 1

        return o_map


#
def cal_angle(current_x, current_y, other_x, other_y):
    p0 = [other_x, other_y]
    p1 = [current_x, current_y]
    p2 = [current_x + 0.1, current_y]
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)
    angle_degree = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
    return angle_degree


def get_circle_occupancy_map(frame_ID, ped_ID, dimensions, neighborhood_radius, grid_radius, grid_angle, data):
    '''
	
    该函数用来计算每一帧中每个行人的圆形占用图
	
    '''
    width, height = dimensions[0], dimensions[1]
    neighborhood_bound = neighborhood_radius / (min(width, height) * 1.0)
    grid_bound = grid_radius / (min(width, height) * 1.0)
    o_map = np.zeros((int(neighborhood_radius / grid_radius), int(360 / grid_angle)))
 

    ped_list = []

    # 搜索同一帧中的所有的行人
    for i in range(len(data[0])):
        if data[0][i] == frame_ID:
            ped_list.append(data[:, i])

    ped_list = np.reshape(ped_list, [-1, 4])
    # 根据目标人与其他人的欧式距离来判断，即欧式距离小于某一设定的阈值，则处于邻域内，否则不在邻域内
    if len(ped_list) == 0:
        print('no pedestrian in this frame!')
    elif len(ped_list) == 1:
        print('only one pedestrian in this frame!')
        return o_map
    else:
        for pedIndex in range(len(ped_list)):
            if ped_list[pedIndex][1] == ped_ID:
                current_x, current_y = ped_list[pedIndex][-1], ped_list[pedIndex][-2]
                current_index = pedIndex
        for otherIndex in range(len(ped_list)):
            if otherIndex != current_index:
                other_x, other_y = ped_list[otherIndex][-1], ped_list[otherIndex][-2]
                other_distance = math.sqrt((other_x - current_x) ** 2 + (other_y - current_y) ** 2)
                angle = cal_angle(current_x, current_y, other_x, other_y)
                if other_distance >= neighborhood_bound:
                    continue
                cell_x = int(np.floor(other_distance / grid_bound))
                cell_y = int(np.floor(angle / grid_angle))

                o_map[cell_x, cell_y] += 1

        return o_map





