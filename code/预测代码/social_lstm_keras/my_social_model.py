import tensorflow as tf
from keras import Input, Model, backend as K
from keras.layers import Dense, Concatenate, Permute
from keras.layers import LSTM
from keras.layers import Lambda, Reshape
from keras.optimizers import RMSprop

from general_utils import pxy_dim, out_dim
from load_model_config import ModelConfig
from tf_normal_sampler import normal2d_log_pdf
from tf_normal_sampler import normal2d_sample
from grid import tf_grid_mask
from general_utils import get_image_size


class MySocialModel:
    def __init__(self, config: ModelConfig) -> None:
        self.x_input = Input((config.obs_len, config.max_n_peds, pxy_dim))
        # y_input = Input((config.obs_len, config.max_n_peds, pxy_dim))
        self.grid_input = Input(
            (config.obs_len, config.max_n_peds, config.max_n_peds,
             config.grid_side_squared))
        self.zeros_input = Input(
            (config.obs_len, config.max_n_peds, config.lstm_state_dim))

        # Social LSTM layers
        self.lstm_layer = LSTM(config.lstm_state_dim, return_state=True)
        self.W_e_relu = Dense(config.emb_dim, activation="relu")
        self.W_a_relu = Dense(config.emb_dim, activation="relu")
        self.W_p = Dense(out_dim)

        self._build_model(config)

    def _compute_loss(self, y_batch, o_batch):
        """
        :param y_batch: (batch_size, pred_len, max_n_peds, pxy_dim)
        :param o_batch: (batch_size, pred_len, max_n_peds, out_dim)
        :return: loss
        """
        not_exist_pid = 0

        y = tf.reshape(y_batch, (-1, pxy_dim))
        o = tf.reshape(o_batch, (-1, out_dim))

        pids = y[:, 0]

        # remain only existing pedestrians data
        exist_rows = tf.not_equal(pids, not_exist_pid)
        y_exist = tf.boolean_mask(y, exist_rows)
        o_exist = tf.boolean_mask(o, exist_rows)
        pos_exist = y_exist[:, 1:]

        # compute 2D normal prob under output parameters
        log_prob_exist = normal2d_log_pdf(o_exist, pos_exist)
        # for numerical stability
        log_prob_exist = tf.minimum(log_prob_exist, 0.0)

        loss = -log_prob_exist
        return loss

    def _compute_social_tensor(self, grid_t, prev_h_t, config):
        """Compute $H_t_i(m, n, :)$.

        this function implementation is same as  getSocialTensor() function.

        :param grid_t: (batch_size, max_n_peds, max_n_peds, grid_side ** 2)
            which is (batch_index, self_pid, other_pid, grid_index).
        :param prev_h_t: (batch_size, max_n_peds, lstm_state_dim)
        :return: H_t (batch_size, max_n_peds, (grid_side ** 2) * lstm_state_dim)
        """
        H_t = []

        for i in range(config.max_n_peds):
            # (batch_size, max_n_peds, max_n_peds, grid_side ** 2)
            # => (batch_size, max_n_peds, grid_side ** 2)
            grid_it = Lambda(lambda grid_t: grid_t[:, i, ...])(grid_t)

            # (batch_size, max_n_peds, grid_side **2)
            # => (batch_size, grid_side ** 2, max_n_peds)
            grid_it_T = Permute((2, 1))(grid_it)

            # (batch_size, grid_side ** 2, lstm_state_dim)
            H_it = Lambda(lambda x: K.batch_dot(x[0], x[1]))(
                [grid_it_T, prev_h_t])

            # store to H_t
            H_t.append(H_it)

        # list of (batch_size, grid_side_squared, lstm_state_dim)
        # => (max_n_peds, batch_size, grid_side_squared, lstm_state_dim)
        H_t = Lambda(lambda H_t: K.stack(H_t, axis=0))(H_t)

        # (max_n_peds, batch_size, grid_side_squared, lstm_state_dim)
        # => (batch_size, max_n_peds, grid_side_squared, lstm_state_dim)
        H_t = Lambda(lambda H_t: K.permute_dimensions(H_t, (1, 0, 2, 3)))(H_t)

        # (batch_size, max_n_peds, grid_side_squared, lstm_state_dim)
        # => (batch_size, max_n_peds, grid_side_squared * lstm_state_dim)
        H_t = Reshape(
            (config.max_n_peds,
             config.grid_side_squared * config.lstm_state_dim))(
            H_t)

        return H_t

    def _build_model(self, config: ModelConfig):
        o_obs_batch = []
        for t in range(config.obs_len):
            print("t: ", t)
            x_t = Lambda(lambda x: x[:, t, :, :])(self.x_input)
            grid_t = Lambda(lambda grid: grid[:, t, ...])(self.grid_input)

            # h_t, c_t = [], []
            # o_t = []

            if t == 0:
                prev_h_t = Lambda(lambda z: z[:, t, :, :])(self.zeros_input)
                prev_c_t = Lambda(lambda z: z[:, t, :, :])(self.zeros_input)

            # compute $H_t$ 计算共享隐藏状态？
            # (n_samples, max_n_peds, (grid_side ** 2) * lstm_state_dim)
            H_t = self._compute_social_tensor(grid_t, prev_h_t, config)

            # for ped_index in range(config.max_n_peds):
            #     print("(t, li):", t, ped_index)
            #     # ----------------------------------------
            #     # compute $e_i^t$ and $a_i^t$
            #     # ----------------------------------------
            #
            #     x_pos_it = Lambda(lambda x_t: x_t[:, ped_index, 1:])(x_t)
            #     e_it = self.W_e_relu(x_pos_it)
            #
            #     # compute a_it
            #     H_it = Lambda(lambda H_t: H_t[:, ped_index, ...])(H_t)
            #     a_it = self.W_a_relu(H_it)
            #
            #     # build concatenated embedding states for LSTM input
            #     emb_it = Concatenate()([e_it, a_it])
            #     emb_it = Reshape((1, 2 * config.emb_dim))(emb_it)
            #
            #     # initial_state = h_i_tになっている
            #     # h_i_tを次のx_t_pに対してLSTMを適用するときのinitial_stateに使えば良い
            #     prev_states_it = [prev_h_t[:, ped_index],
            #                       prev_c_t[:, ped_index]]
            #     lstm_output, h_it, c_it = self.lstm_layer(emb_it,
            #                                               prev_states_it)
            #
            #     h_t.append(h_it)
            #     c_t.append(c_it)
            #
            #     # compute output_it, which shape is (batch_size, 5)
            #     o_it = self.W_p(lstm_output)
            #     o_t.append(o_it)

            prev_ht_ct_ot = [prev_h_t, prev_c_t, x_t]
            h_t, c_t, o_t, = self.compute_ht_ct_ot(config, prev_ht_ct_ot, H_t, t, is_pred=False)

            # convert lists of h_it/c_it/o_it to h_t/c_t/o_t respectively
            # h_t = _stack_permute_axis_zero(h_t)
            # c_t = _stack_permute_axis_zero(c_t)
            # o_t = _stack_permute_axis_zero(o_t)

            o_obs_batch.append(o_t)

            # current => previous
            prev_h_t = h_t
            prev_c_t = c_t


        # convert list of output_t to output_batch
        o_obs_batch = _stack_permute_axis_zero(o_obs_batch)

        # ----------------------------------------------------------------------
        # Prediction 位置估计？
        # ----------------------------------------------------------------------
        # この時点でprev_h_t, prev_c_tにはobs_lenの最終的な状態が残っている

        x_obs_t_final = Lambda(lambda x: x[:, -1, :, :])(self.x_input)
        pid_obs_t_final = Lambda(lambda x_t: x_t[:, :, 0])(x_obs_t_final)
        pid_obs_t_final = Lambda(lambda p_t: K.expand_dims(p_t, 2))(
            pid_obs_t_final)

        x_pred_batch = []
        o_pred_batch = []
        for t in range(config.pred_len):
            if t == 0:
                prev_o_t = Lambda(lambda o_b: o_b[:, -1, :, :])(o_obs_batch)
            # 坐标预测根据双变量正太分布采样
            pred_pos_t = normal2d_sample(prev_o_t)
            # assume all the pedestrians in the final observation frame are
            # exist in the prediction frames： 假设最后一个观测帧中的所有行人都存在于预测帧中。
            x_pred_t = Concatenate(axis=2)([pid_obs_t_final, pred_pos_t])

            grid_t = tf_grid_mask(x_pred_t, get_image_size(config.test_dataset_kind),
                                  config.n_neighbor_pixels, config.grid_side)

            # h_t, c_t, o_t = [], [], []

            # compute $H_t$
            # (n_samples, max_n_peds, (grid_side ** 2) * lstm_state_dim)
            H_t = self._compute_social_tensor(grid_t, prev_h_t, config)

            prev_ht_ct_ot = [prev_h_t, prev_c_t, prev_o_t]
            # for i in range(config.max_n_peds):
            #     print("(t, li):", t, i)
            #
            #     prev_o_it = Lambda(lambda o_t: o_t[:, i, :])(prev_o_t)
            #     H_it = Lambda(lambda H_t: H_t[:, i, ...])(H_t)
            #
            #     # pred_pos_it: (batch_size, 2)
            #     pred_pos_it = normal2d_sample(prev_o_it)
            #
            #     # compute e_it and a_it
            #     # e_it: (batch_size, emb_dim)
            #     # a_it: (batch_size, emb_dim)
            #     e_it = self.W_e_relu(pred_pos_it)
            #     a_it = self.W_a_relu(H_it)
            #
            #     # build concatenated embedding states for LSTM input
            #     # emb_it: (batch_size, 1, 2 * emb_dim)
            #     emb_it = Concatenate()([e_it, a_it])
            #     emb_it = Reshape((1, 2 * config.emb_dim))(emb_it)
            #
            #     # initial_state = h_i_tになっている
            #     # h_i_tを次のx_t_pに対してLSTMを適用するときのinitial_stateに使えば良い
            #     prev_states_it = [prev_h_t[:, i], prev_c_t[:, i]]
            #     lstm_output, h_it, c_it = self.lstm_layer(emb_it,
            #                                               prev_states_it)
            #
            #     h_t.append(h_it)
            #     c_t.append(c_it)
            #
            #     # compute output_it, which shape is (batch_size, 5)
            #     o_it = self.W_p(lstm_output)
            #     o_t.append(o_it)

            h_t, c_t, o_t, = self.compute_ht_ct_ot(config, prev_ht_ct_ot, H_t, t)
            # convert lists of h_it/c_it/o_it to h_t/c_t/o_t respectively
            # h_t = _stack_permute_axis_zero(h_t)
            # c_t = _stack_permute_axis_zero(c_t)
            # o_t = _stack_permute_axis_zero(o_t)

            o_pred_batch.append(o_t)
            x_pred_batch.append(x_pred_t)

            # current => previous
            prev_h_t = h_t
            prev_c_t = c_t
            prev_o_t = o_t

        # convert list of output_t to output_batch
        o_pred_batch = _stack_permute_axis_zero(o_pred_batch)
        x_pred_batch = _stack_permute_axis_zero(x_pred_batch)

        # o_concat_batch = Lambda(lambda os: tf.concat(os, axis=1))(
        #     [o_obs_batch, o_pred_batch])

        # 本当に学習に必要なモデルはこっちのはず
        self.train_model = Model(
            [self.x_input, self.grid_input, self.zeros_input],
            o_pred_batch
        )

        lr = 0.003
        optimizer = RMSprop(lr=config.lr)
        self.train_model.compile(optimizer, self._compute_loss)

        self.sample_model = Model(
            [self.x_input, self.grid_input, self.zeros_input],
            x_pred_batch
        )

    def compute_ht_ct_ot(self, config, prev_ht_ct_ot, H_t, t, is_pred=True):

        prev_h_t, prev_c_t, prev_o_t = prev_ht_ct_ot
        h_t, c_t, o_t = [], [], []
        for ped_index in range(config.max_n_peds):
            print("(t, ped_index):", t, ped_index)

            # compute H_it
            H_it = Lambda(lambda H_t: H_t[:, ped_index, ...])(H_t)

            # pred_pos_it: (batch_size, 2)
            if is_pred:
                prev_o_it = Lambda(lambda o_t: o_t[:, ped_index, :])(prev_o_t)
                pred_pos_it = normal2d_sample(prev_o_it)
            else:
                prev_o_it = Lambda(lambda o_t: o_t[:, ped_index, 1:])(prev_o_t)
                pred_pos_it = prev_o_it

            # compute e_it and a_it
            # e_it: (batch_size, emb_dim)
            # a_it: (batch_size, emb_dim)
            e_it = self.W_e_relu(pred_pos_it)
            a_it = self.W_a_relu(H_it)

            # build concatenated embedding states for LSTM input
            # emb_it: (batch_size, 1, 2 * emb_dim)
            emb_it = Concatenate()([e_it, a_it])
            emb_it = Reshape((1, 2 * config.emb_dim))(emb_it)

            # initial_state = h_i_tになっている
            # h_i_tを次のx_t_pに対してLSTMを適用するときのinitial_stateに使えば良い
            prev_states_it = [prev_h_t[:, ped_index], prev_c_t[:, ped_index]]
            lstm_output, h_it, c_it = self.lstm_layer(emb_it,
                                                      prev_states_it)

            h_t.append(h_it)
            c_t.append(c_it)

            # compute output_it, which shape is (batch_size, 5)
            o_it = self.W_p(lstm_output)
            o_t.append(o_it)

        h_t = _stack_permute_axis_zero(h_t)
        c_t = _stack_permute_axis_zero(c_t)
        o_t = _stack_permute_axis_zero(o_t)
        return h_t, c_t, o_t

def _stack_permute_axis_zero(xs):
    xs = Lambda(lambda xs: K.stack(xs, axis=0))(xs)

    # axes (0, 1) are permuted
    perm = [1, 0] + list(range(2, xs.shape.ndims))
    xs = Lambda(lambda xs: K.permute_dimensions(xs, perm))(xs)
    return xs
