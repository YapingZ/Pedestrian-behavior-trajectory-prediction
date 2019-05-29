import json


class ModelConfig:
    def __init__(self, n_epochs, batch_size, obs_len, pred_len,
                 max_n_peds, n_neighbor_pixels, grid_side, lstm_state_dim,
                 emb_dim, data_root, test_dataset_kind, steps_per_epoch, lr, **kwargs):
        # train config
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # model config
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.max_n_peds = max_n_peds
        self.n_neighbor_pixels = n_neighbor_pixels
        self.grid_side = grid_side
        self.grid_side_squared = grid_side ** 2

        # layer config
        self.lstm_state_dim = lstm_state_dim
        self.emb_dim = emb_dim
        self.steps_per_epoch= steps_per_epoch
        self.lr = lr

        # dataset config
        self.data_root = data_root
        self.test_dataset_kind = test_dataset_kind


def load_model_config(config_file: str) -> ModelConfig:
    with open(config_file, "r") as f:
        config = json.load(f)

    return ModelConfig(**config)
