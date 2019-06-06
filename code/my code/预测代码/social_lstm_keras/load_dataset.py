from functools import reduce

import numpy as np

from general_utils import get_data_dir
from general_utils import get_image_size
from grid import grid_mask
from load_model_config import ModelConfig
from preprocessors.preprocessors_utils import create_dataset_preprocessor


def load_dataset_from_config(config: ModelConfig):
    data_dir = get_data_dir(config.data_root, config.test_dataset_kind)

    return load_dataset(data_dir=data_dir,
                        dataset_kind=config.test_dataset_kind, config=config)


def load_dataset(data_dir, dataset_kind, config):
    loader = _SingleDatasetLoader(data_dir, dataset_kind, config)
    dataset = loader.load()
    return dataset


class SingleDataset:
    def __init__(self, frame_data, config, image_size):
        self.seq_len = config.obs_len + config.pred_len
        self.max_n_peds = config.max_n_peds
        self.n_neighbor_pixels = config.n_neighbor_pixels
        self.grid_side = config.grid_side
        self.image_size = image_size
        self.config = config

        self.x_data, self.y_data, self.grid_data = self._build_data(frame_data)

    def _build_data(self, frame_data):
        x_data = []
        y_data = []

        for i in range(len(frame_data) - self.seq_len):
            cf_data = frame_data[i:i + self.seq_len, ...]
            nf_data = frame_data[i + 1:i + self.seq_len + 1, ...]

            ped_col_index = 0
            # 将当前序列中行人的id和下个序列中行人的id分开保存
           
            cf_ped_ids = reduce(set.intersection,
                                [set(nf_ped_ids) for nf_ped_ids in
                                 cf_data[..., ped_col_index]])

            nf_ped_ids = reduce(set.intersection,
                                [set(nf_ped_ids) for nf_ped_ids in
                                 nf_data[..., ped_col_index]])

            ped_ids = list(cf_ped_ids & nf_ped_ids - {0})
            # 判断当前时刻是否有行人
            if not ped_ids:
                continue

            x = np.zeros((self.seq_len, self.max_n_peds, 3))
            y = np.zeros((self.seq_len, self.max_n_peds, 3))

            # fi 表示帧的索引号, cf 表示当前帧, nf 表示下一帧
            for fi, (cf, nf) in enumerate(zip(cf_data, nf_data)):
                for j, ped_id in enumerate(ped_ids):
                    cf_ped_row = cf[:, 0] == ped_id
                    nf_ped_row = nf[:, 0] == ped_id

                    if np.any(cf_ped_row):
                        x[fi, j, :] = cf[cf[:, 0] == ped_id]
                    if np.any(nf_ped_row):
                        y[fi, j, :] = nf[nf[:, 0] == ped_id]

            x_data.append(x)
            y_data.append(y)

        
        grid_data = [grid_mask(x, self.image_size, self.config) for x in x_data]

        data_tuple = (np.array(x_data, np.float32),
                      np.array(y_data, np.float32),
                      np.array(grid_data, np.float32))
        return data_tuple

    def get_data(self, lstm_state_dim):
        zeros_data = np.zeros((len(self.x_data), self.seq_len,
                               self.max_n_peds, lstm_state_dim), np.float32)

        return self.x_data, self.y_data, self.grid_data, zeros_data


class _SingleDatasetLoader:
    def __init__(self, data_dir, dataset_kind, config):

        self.data_dir = data_dir
        
        self.dataset_kind = dataset_kind
        self.image_size = get_image_size(dataset_kind)

        self.config = config

    def load(self) -> SingleDataset:
        preprocessor = create_dataset_preprocessor(
            self.data_dir, self.dataset_kind)
        df = preprocessor.preprocess_frame_data()

        # 当前数据库中的所有帧的行人ID
        all_frames = df["frame"].unique().tolist()
        n_all_frames = len(all_frames)

        all_frame_data = np.zeros((n_all_frames, self.config.max_n_peds, 3),
                                  np.float64)
        for index, frame in enumerate(all_frames):
            peds_with_pos = np.array(df[df["frame"] == frame][["id", "x", "y"]])

            n_peds = len(peds_with_pos)

            all_frame_data[index, 0:n_peds, :] = peds_with_pos

        return SingleDataset(all_frame_data, self.config, self.image_size)
