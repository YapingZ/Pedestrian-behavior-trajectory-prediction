import numpy as np


def _to_normal_params(output5d):
    x_mean, y_mean = output5d[0], output5d[1]
    x_std, y_std = np.exp(output5d[2]), np.exp(output5d[3])
    cor = np.tanh(output5d[4])

    return x_mean, y_mean, x_std, y_std, cor


def sample_normal2d(output5d):
    """Sample a 2D point.
    :param output5d:
    :return:
    >>> output5d = np.arange(5)
    >>> s = sample_normal2d(output5d)
    >>> s.shape
    (1, 2)
    """
    x_mean, y_mean, x_std, y_std, cor = _to_normal_params(output5d)

    mean = np.array([x_mean, y_mean], np.float32)

    x_var = np.square(x_std)
    y_var = np.square(y_std)
    xy_cor = x_std * y_std * cor
    cov = np.array([[x_var, xy_cor], [xy_cor, y_var]], np.float32)

    return np.random.multivariate_normal(mean, cov, 1)
