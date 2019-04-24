import os

import numpy as np
import pandas as pd

from general_utils import get_image_size


class EthDatasetPreprosessor:
    """Preprocessor for ETH dataset.

    """

    def __init__(self, data_dir, dataset_kind):
        self._data_dir = data_dir
        self.image_size = get_image_size(dataset_kind)

        pass

    def preprocess_frame_data(self):
        # load homography matrix and raw pedestrians trajectories
        homography_file = os.path.join(self.data_dir, "H.txt")
        obsmat_file = os.path.join(self.data_dir, "obsmat.txt")

        H = np.genfromtxt(homography_file)
        obs_columns = ["frame", "id", "px", "pz", "py", "vx", "vz", "vy"]
        obs_df = pd.DataFrame(np.genfromtxt(obsmat_file), columns=obs_columns)

        # remain only (frame index, pedestrian id, position x, position y)
        pos_df_raw = obs_df[["frame", "id", "px", "py"]]

        # ----------------------
        # position preprocessing
        # ----------------------
        xy = np.array(pos_df_raw[["px", "py"]])

        # world xy to image xy: inverse mapping of homography
        xy = self._world_to_image_xy(xy, H)

        # normalize
        xy = xy / self.image_size

        # construct preprocessed df
        pos_df_preprocessed = pd.DataFrame({
            "frame": pos_df_raw["frame"],
            "id": pos_df_raw["id"],
            "x": xy[:, 0],
            "y": xy[:, 1]
        })

        return pos_df_preprocessed

    @property
    def data_dir(self):
        return self._data_dir

    @staticmethod
    def _world_to_image_xy(world_xy, H):
        """Convert world (x, y) position to image (x, y) position.

        This function use inverse mapping of homography transform.

        :param world_xy: world (x, y) positions
        :param H: homography matrix
        :return: image (x, y) positions
        """
        world_xy = np.array(world_xy)
        world_xy1 = np.concatenate([world_xy, np.ones((len(world_xy), 1))],
                                   axis=1)
        image_xy1 = np.linalg.inv(H).dot(world_xy1.T).T
        image_xy = image_xy1[:, :2] / np.expand_dims(image_xy1[:, 2], axis=1)
        return image_xy
