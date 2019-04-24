import keras.backend as K
import tensorflow as tf
import tensorflow.contrib.distributions as ds
from keras.layers import Multiply
from keras.layers import Reshape, Lambda, Concatenate


def _to_normal2d(output_batch) -> ds.MultivariateNormalTriL:
    """
    :param output_batch: (n_samples, 5)
    :return
    """

    # mean of x and y
    x_mean = Lambda(lambda o: o[:, 0])(output_batch)
    y_mean = Lambda(lambda o: o[:, 1])(output_batch)

    # std of x and y
    # std is must be 0 or positive
    x_std = Lambda(lambda o: K.exp(o[:, 2]))(output_batch)
    y_std = Lambda(lambda o: K.exp(o[:, 3]))(output_batch)

    # correlation coefficient
    # correlation coefficient range is [-1, 1]
    cor = Lambda(lambda o: K.tanh(o[:, 4]))(output_batch)

    loc = Concatenate()([
        Lambda(lambda x_mean: K.expand_dims(x_mean, 1))(x_mean),
        Lambda(lambda y_mean: K.expand_dims(y_mean, 1))(y_mean)
    ])

    x_var = Lambda(lambda x_std: K.square(x_std))(x_std)
    y_var = Lambda(lambda y_std: K.square(y_std))(y_std)
    xy_cor = Multiply()([x_std, y_std, cor])

    cov = Lambda(lambda inputs: K.stack(inputs, axis=0))(
        [x_var, xy_cor, xy_cor, y_var])
    cov = Lambda(lambda cov: K.permute_dimensions(cov, (1, 0)))(cov)
    cov = Reshape((2, 2))(cov)

    scale_tril = Lambda(lambda cov: tf.cholesky(cov))(cov)
    mvn = ds.MultivariateNormalTriL(loc, scale_tril)

    return mvn


def normal2d_log_pdf(output_batch, pos_batch):
    """
    :param output_batch (n_samples, 5):
    :param pos_batch (n_samples, 2):
    :return: (n_samples,)
    """
    mvn = _to_normal2d(output_batch)
    log_prob_batch = Lambda(lambda pos: mvn.log_prob(pos))(pos_batch)
    return log_prob_batch


def normal2d_sample(output_batch):
    """
    :param output_batch: (..., 5)
    :return: (..., 2)
    """
    original_output_shape = output_batch.shape
    o = Lambda(lambda o: tf.reshape(o, (-1, 5)))(output_batch)
    sample = Lambda(lambda o: _to_normal2d(o).sample())(o)

    expected_sample_shape = original_output_shape[:-1].concatenate(
        sample.shape[-1]).as_list()

    if expected_sample_shape[0] is None:
        expected_sample_shape[0] = -1

    return Lambda(lambda s: tf.reshape(s, expected_sample_shape))(sample)
