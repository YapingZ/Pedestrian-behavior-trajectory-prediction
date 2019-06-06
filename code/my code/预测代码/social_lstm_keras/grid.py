from functools import partial

import numpy as np
import tensorflow as tf
from keras.layers import Lambda
from keras.utils import to_categorical


def tf_grid_mask(x, image_size, n_neighbor_pixels, grid_side):
    func = partial(_tf_grid_mask_frame, image_size=image_size,
                   n_neighbor_pixels=n_neighbor_pixels, grid_side=grid_side)

    return Lambda(
        lambda x: tf.stack(tf.map_fn(lambda frame: func(frame), x)))(x)


def grid_mask(x, image_size, config):
    func = partial(_grid_mask_frame, image_size=image_size, config=config)

    return np.array([func(frame) for frame in x])


def _tf_grid_mask_frame(frame, image_size, n_neighbor_pixels, grid_side):
   
    max_n_peds = frame.shape.as_list()[0]
    pids = frame[:, 0]

    

    def compute_id_mask(pids):
        id_mask = tf.tensordot(tf.expand_dims(pids, axis=1),
                               tf.transpose(tf.expand_dims(pids, axis=1)),
                               axes=(1, 0))
        id_mask = tf.cast(id_mask, tf.bool)
        
        id_mask = tf.logical_and(
            tf.logical_not(tf.cast(tf.eye(max_n_peds), tf.bool)), id_mask)
        id_mask = tf.expand_dims(id_mask, axis=2)
        id_mask = tf.cast(id_mask, tf.float32)

        return id_mask

    id_mask = Lambda(compute_id_mask)(pids)

    bound = n_neighbor_pixels / np.array(image_size)
    pos = frame[:, 1:]
    tl = pos - bound / 2
    br = pos + bound / 2

    frame_mask = []
    for self_index in range(max_n_peds):
        is_neighbor = Lambda(lambda pos: tf.cast(tf.reduce_all(
            tf.concat([tl[self_index] <= pos, pos < br[self_index]], axis=1),
            axis=1), np.int32))(pos)

        cell_xy = Lambda(lambda pos: tf.cast(
            tf.floor(((pos - tl[self_index]) / bound) * grid_side), tf.int32))(
            pos)

        cell_index = cell_xy[:, 0] + cell_xy[:, 1] * grid_side
        cell_index = cell_index * is_neighbor

        self_frame_mask = tf.stack(tf.map_fn(
            lambda c: tf.eye(grid_side ** 2, dtype=np.int32)[c], cell_index),
            axis=0)
        self_frame_mask *= tf.expand_dims(is_neighbor, 1)
        frame_mask.append(self_frame_mask)

    frame_mask = tf.stack(frame_mask, axis=0)
    frame_mask = tf.cast(frame_mask, tf.float32)
  
    frame_mask *= id_mask
    return frame_mask


def _grid_mask_frame(frame, image_size, config):
    """
    此函数计算表示的二进制掩码
 
    
    image_size : 图像尺寸
    n_neighbor_pixels : 邻域
    grid_side : 标量值表示网格离散化的大小
    """
    # 保留序列最长的行人
    max_n_peds = frame.shape[0]

   
    pids = frame[:, 0]
    id_mask = np.dot(np.expand_dims(pids, axis=1),
                     np.expand_dims(pids, axis=1).T)
    id_mask[id_mask > 0] = 1
   
    id_mask *= ~np.eye(max_n_peds).astype(np.bool)
    id_mask = np.expand_dims(id_mask, axis=2)

    # bound = config.n_neighbor_pixels / np.array(image_size)
    pos = frame[:, 1:]
    # tl = pos - bound / 2
    # br = pos + bound / 2

    frame_mask = []
    for self_index in range(max_n_peds):
        # is_neighbor = np.all(
        #     np.concatenate([tl[self_index] <= pos, pos < br[self_index]],
        #                    axis=1),
        #     axis=1)

        if config.grid_type == 'circle':
            cell_xy, is_neighbor = is_neighbor_circle(
                pos, self_index, image_size, config.n_neighbor_pixels)
        else:
            cell_xy, is_neighbor = is_neighbor_rect(
                pos, self_index, image_size, config.n_neighbor_pixels, config.grid_side)


        # cell_xy = np.floor(((pos - tl[self_index]) / bound) * grid_side).astype(
        #     np.int)
        cell_index = cell_xy[:, 0] + cell_xy[:, 1] * config.grid_side
        cell_index *= is_neighbor

        self_frame_mask = to_categorical(cell_index, config.grid_side ** 2)
        self_frame_mask *= np.expand_dims(is_neighbor, 1)
        frame_mask.append(self_frame_mask)

    frame_mask = np.stack(frame_mask, axis=0)
  
    frame_mask *= id_mask


    return frame_mask


def is_neighbor_rect(pos, self_index, image_size, n_neighbor_pixels=32, grid_side=4):
    # 计算极坐标
    bound = n_neighbor_pixels / np.array(image_size)
    tl = pos - bound / 2
    br = pos + bound / 2
    is_neighbor = np.all(np.concatenate(
        [tl[self_index] <= pos, pos < br[self_index]], axis=1), axis=1)
    cell_xy = np.floor(((pos - tl[self_index]) / bound) * grid_side).astype(np.int)
    return cell_xy, is_neighbor

def is_neighbor_circle(pos, self_index, image_size, grid_radius=4, grid_angle=45, n_neighbor_pixels=32):
    """
    判断是否在圆形邻域内，并计算变换坐标
    :param pos: 当前帧所有人坐标
    :param self_index: 当前目标的索引
    :param image_size: 图像尺寸
    :param grid_radius: 网格大小
    :param grid_angle:网格角度
    :param n_neighbor_pixels: 邻域大小
    :return:
    """
    # 获取输入图像的宽度和高度
    width, height = int(image_size[0]), int(image_size[1])
    # 设置邻域判断阈值
    neighborhood_bound = n_neighbor_pixels / (min(width, height) * 1.0)
    grid_bound = grid_radius / (min(width, height) * 1.0)
    # 获取目标（人）当前坐标
    current_x, current_y = pos[self_index]
    # 获取所有人的坐标
    other_x, other_y = pos[:, 0], pos[:, 1]
    # 计算目标与其他所有人的欧式距离
    other_distance = np.sqrt((other_x - current_x) ** 2 + (other_y - current_y) ** 2)
    # 计算角度
    angle = np.array(
        [cal_angle(current_x, current_y, other_x[indx], other_y[indx]) for indx in range(len(other_x))])
    # 计算坐标偏差
    cell_x = np.floor(other_distance / grid_bound)
    # 计算极坐标
    cell_y = np.floor(angle / grid_angle)
    cell_xy = np.concatenate((np.expand_dims(cell_x, -1), np.expand_dims(cell_y, -1)), 1).astype(
            np.int)
    # 判断其他人是否在邻域内
    is_neighbor = other_distance < neighborhood_bound

    return cell_xy, is_neighbor


def cal_angle(current_x, current_y, other_x, other_y):
    p0 = [other_x, other_y]
    p1 = [current_x, current_y]
    p2 = [current_x + 0.1, current_y]
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)
    angle_degree = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
    return angle_degree

def _legacy_grid_mask_frame(frame, image_size, n_neighbor_pixels, grid_side):
    max_n_peds = frame.shape[0]
    width, height = image_size[0], image_size[1]

    frame_mask = np.zeros((max_n_peds, max_n_peds, grid_side ** 2))

    width_bound = n_neighbor_pixels / width
    height_bound = n_neighbor_pixels / height

    # 遍历当前帧中的每个行人序列
    for self_index in range(max_n_peds):
        # 如果pedID为零，则不存在行人
        if frame[self_index, 0] == 0:
            # 对于不存在的行人，二进制掩码为零
            continue

        # 获取当前行人的x和y坐标
        self_x, self_y = frame[self_index, 1], frame[self_index, 2]

        tl_x, br_x = self_x - width_bound / 2, self_x + width_bound / 2
        tl_y, br_y = self_y - height_bound / 2, self_y + height_bound / 2

        # 遍历所有的行人
        for other_index in range(max_n_peds):
            
            if frame[other_index, 0] == 0:
               
                continue

           
            if frame[other_index, 0] == frame[self_index, 0]:
           
                continue

            # 获得另一个行人的x和y坐标
            other_x, other_y = frame[other_index, 1], frame[other_index, 2]
            if other_x >= br_x or other_x < tl_x \
                    or other_y >= br_y or other_y < tl_y:
                # 如果目标行人周围没有干扰物，则二元掩模为零
                continue

            # 如果周围有别的行人，计算与目标行人的距离
            cell_x = int(np.floor(((other_x - tl_x) / width_bound) * grid_side))
            cell_y = int(
                np.floor(((other_y - tl_y) / height_bound) * grid_side))

            
            frame_mask[self_index, other_index, cell_x + cell_y * grid_side] = 1

    return frame_mask
