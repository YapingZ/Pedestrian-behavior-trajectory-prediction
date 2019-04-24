import numpy as np


def compute_abe(x_true, x_pred):
    """Compute Average displacement error (ade).

    In the original paper, ade is mean square error (mse) over all estimated
    points of a trajectory and the true points.

    :param x_true: (n_samples, seq_len, max_n_peds, 3)
    :param x_pred: (n_samples, seq_len, max_n_peds, 3)
    :return: Average displacement error
    """
    # pid != 0 means there is the person at the frame.
    not_exist_pid = 0
    exist_elements = x_true[..., 0] != not_exist_pid

    # extract pedestrians positions (x, y), then compute difference
    pos_true = x_true[..., 1:]
    pos_pred = x_pred[..., 1:]
    diff = pos_true - pos_pred

    # ade = average displacement error
    ade = np.mean(np.square(diff[exist_elements]))
    return ade


def compute_fde(x_true, x_pred):
    """Compute Final displacement error (fde).

    In the original paper, ade is mean square error (mse) over all estimated
    points of a trajectory and the true points.

    :param x_true: (n_samples, seq_len, max_n_peds, 3)
    :param x_pred: (n_samples, seq_len, max_n_peds, 3)
    :return: Average displacement error
    """
    # pid != 0 means there is the person at the frame.
    not_exist_pid = 0
    exist_final_elements = x_true[:, -1, :, 0] != not_exist_pid

    # extract pedestrians positions (x, y), then compute difference
    pos_final_true = x_true[:, -1, :, 1:]
    pos_final_pred = x_pred[:, -1, :, 1:]
    diff = pos_final_true - pos_final_pred

    fde = np.mean(np.square(diff[exist_final_elements]))
    return fde
