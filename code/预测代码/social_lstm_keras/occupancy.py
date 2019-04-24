


import numpy as np
import math


def get_rectangular_occupancy_map(frame_ID, ped_ID, dimensions, neighborhood_size, grid_size, data):
    """
    This function computes rectangular occupancy map for each pedestrian at each frame.
    This occupancy map is used in group level LSTM.
    params:
        frame_ID: frame No.
        ped_ID: each ped in frame_ID
        dimensions : This will be a list [width, height], size of frame
        neighborhood_size : Scalar value representing the size of neighborhood considered (32)
        grid_size : Scalar value representing the size of the grid discretization (4)
        data: data of pixel_pos.csv file, [frame_ID, ped_ID, y-coord, x-coord]
    """
    #    width_bound, height_bound = neighborhood_size/(width*1.0), neighborhood_size/(height*1.0)
    #    width_grid_bound, height_grid_bound = grid_size/(width*1.0), grid_size/(height*1.0)

    o_map = np.zeros((int(neighborhood_size / grid_size), int(neighborhood_size / grid_size)))
    #    o_map = np.zeros((int(neighborhood_size/grid_size)**2))

    ped_list = []

    # search for all peds in the same frame
    for i in range(len(data[0])):
        if data[0][i] == frame_ID:
            ped_list.append(data[:, i])
    # reshape ped_list to [num of ped, 4], [frame_ID, ped_ID, y-coord, x-coord]
    ped_list = np.reshape(ped_list, [-1, 4])

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
    This function computes rectangular occupancy map for each pedestrian at each frame.
    This occupancy map is used in group level LSTM.
    params:
        frame_ID: frame No.
        ped_ID: each ped in frame_ID
        dimensions : This will be a list [width, height], size of frame
        neighborhood_size : Scalar value representing the size of neighborhood considered (32)
        grid_size : Scalar value representing the size of the grid discretization (4)
        data: data of pixel_pos.csv file, [frame_ID, ped_ID, y-coord, x-coord]
    '''
    width, height = dimensions[0], dimensions[1]
    neighborhood_bound = neighborhood_radius / (min(width, height) * 1.0)
    grid_bound = grid_radius / (min(width, height) * 1.0)
    o_map = np.zeros((int(neighborhood_radius / grid_radius), int(360 / grid_angle)))
    #    o_map = np.zeros((int(neighborhood_size/grid_size)**2))

    ped_list = []

    # search for all peds in the same frame
    for i in range(len(data[0])):
        if data[0][i] == frame_ID:
            ped_list.append(data[:, i])
    # reshape ped_list to [num of ped, 4], [frame_ID, ped_ID, y-coord, x-coord]
    ped_list = np.reshape(ped_list, [-1, 4])

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


def log_circle_occupancy_map(frame_ID, ped_ID, dimensions, neighborhood_radius, grid_radius, grid_angle, data):
    """
    This function computes occupancy map for each pedestrian at each frame.
    This occupancy map is used in group level LSTM.
    params:
        frame_ID: frame No.
        ped_ID: each ped in frame_ID
        dimensions : This will be a list [width, height], size of frame
        neighborhood_size : Scalar value representing the size of neighborhood considered (32)
        grid_size : Scalar value representing the size of the grid discretization (4)
        data: data of pixel_pos.csv file, [frame_ID, ped_ID, y-coord, x-coord]
    """
    width, height = dimensions[0], dimensions[1]
    o_map = np.zeros((8, 8))
    #    o_map = np.zeros((int(neighborhood_size/grid_size)**2))

    ped_list = []

    # search for all peds in the same frame
    for i in range(len(data[0])):
        if data[0][i] == frame_ID:
            ped_list.append(data[:, i])
    # reshape ped_list to [num of ped, 4], [frame_ID, ped_ID, y-coord, x-coord]
    ped_list = np.reshape(ped_list, [-1, 4])

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
                other_distance = math.sqrt(
                    (other_x * width - current_x * width) ** 2 + (other_y * height - current_y * height) ** 2)
                log_distance = math.log2(other_distance)
                angle = cal_angle(current_x, current_y, other_x, other_y)
                if other_distance >= 8:
                    continue
                cell_x = int(np.floor(log_distance))
                cell_y = int(np.floor(angle / grid_angle))

                o_map[cell_x, cell_y] += 1

        return o_map

