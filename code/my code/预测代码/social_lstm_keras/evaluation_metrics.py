import numpy as np


def compute_ade(x_true, x_pred):
    """计算平均位移误差 (ade).

    在原始论文中, ade 表示的是均方误差 (mse) 
    

    :param x_true: 真实点
    :param x_pred: 预测的轨迹点
    :return: 返回平均位移误差 (ade)
    """
    # pid != 0 代表当前帧中有行人
    not_exist_pid = 0
    exist_elements = x_true[..., 0] != not_exist_pid

    # 提取行人位置（x，y），然后计算差值
    pos_true = x_true[..., 1:]
    pos_pred = x_pred[..., 1:]
    diff = pos_true - pos_pred

    # 计算平均位移误差
    ade = np.mean(np.square(diff[exist_elements]))
    return ade


def compute_fde(x_true, x_pred):
    """计算最终位移误差 (fde).


    :param x_true: 真实点
    :param x_pred: 预测的轨迹点
    :return: 返回最终位移误差 (fde)
    """
    # pid != 0 代表当前帧中有行人
    not_exist_pid = 0
    exist_final_elements = x_true[:, -1, :, 0] != not_exist_pid

    # 提取行人最终位置（x，y），然后计算差值
    pos_final_true = x_true[:, -1, :, 1:]
    pos_final_pred = x_pred[:, -1, :, 1:]
    diff = pos_final_true - pos_final_pred

    fde = np.mean(np.square(diff[exist_final_elements]))
    return fde
