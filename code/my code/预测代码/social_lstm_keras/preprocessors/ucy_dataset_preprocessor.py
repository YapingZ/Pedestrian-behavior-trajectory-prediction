import os

import numpy as np
import pandas as pd

from general_utils import get_image_size


class UcyDatasetPreprocessor:
    ped_start_line_words = "Num of control points"
    vsp_columns = ["x", "y", "frame", "gaze", "dummy1", "dummy2", "dummy3",
                   "dummy4"]

    # arrange frame interval
    frame_interval = 10

    def __init__(self, data_dir, dataset_kind):
        self._data_dir = data_dir
        self._dataset_kind = dataset_kind

    def preprocess_frame_data(self):
        lines = self._read_lines(self._get_vsp_file(self._data_dir))

        # find first lines specifying each pedestrian trajectory
        ped_start_indices = [li for li, line in enumerate(lines) if
                             line.find(self.ped_start_line_words) != -1]

        pos_df_raw = []

        # extract pedestrian positions as a data frame
        for i, start_index in enumerate(ped_start_indices):
            n_pos_i = int(lines[start_index].split()[0])
            pos_lines_i = lines[start_index + 1:start_index + 1 + n_pos_i]
            pos_df_raw_i = pd.DataFrame([line.split() for line in pos_lines_i],
                                        columns=self.vsp_columns)
            # in UCY dataset, pedestiran "id" is not given,
            # therefore add "id" column with serial number.
            pos_df_raw_i["id"] = i + 1

            pos_df_raw.append(pos_df_raw_i)

        pos_df_raw = pd.concat(pos_df_raw)
        # remain only (frame, id, x, y)
        pos_df_raw = pos_df_raw[["frame", "id", "x", "y"]].astype(np.float32)
        pos_df_raw = pos_df_raw.reset_index(drop=True)

        # interpolate & normalize & thin out
        pos_df_preprocessed = self.interpolate_pos_df(pos_df_raw)
        pos_df_preprocessed = self.normalize_pos_df(pos_df_preprocessed,
                                                    self._dataset_kind)
        pos_df_preprocessed = self.thin_out_pos_df(pos_df_preprocessed,
                                                   self.frame_interval)
        pos_df_preprocessed = pos_df_preprocessed.sort_values(["frame", "id"])

        return pos_df_preprocessed

    @staticmethod
    def _get_vsp_file(data_dir):
        vsp_file_name = "{}.vsp".format(os.path.basename(data_dir))
        vsp_file = os.path.join(data_dir, vsp_file_name)

        return vsp_file

    @staticmethod
    def _read_lines(file):
        with open(file, "r") as f:
            return f.readlines()

    @staticmethod
    def interpolate_pos_df(pos_df):
        pos_df_interp = []

        for pid, pid_df in pos_df.groupby("id"):
            observed_frames = np.array(pid_df["frame"])
            frame_range = np.arange(observed_frames[0], observed_frames[-1] + 1)

            x_interp = np.interp(frame_range, pid_df["frame"], pid_df["x"])
            y_interp = np.interp(frame_range, pid_df["frame"], pid_df["y"])

            pos_df_interp.append(pd.DataFrame({
                "frame": frame_range,
                "id": pid,
                "x": x_interp,
                "y": y_interp
            }))

        pos_df_interp = pd.concat(pos_df_interp)
        return pos_df_interp

    @staticmethod
    def normalize_pos_df(pos_df, dataset_kind):
        image_size = np.array(get_image_size(dataset_kind))

        xy = np.array(pos_df[["x", "y"]])
        # originally (0, 0) is the center of the frame,
        # therefore move (0, 0) to top-left
        xy += image_size / 2
        # clipping
        xy[:, 0] = np.clip(xy[:, 0], 0.0, image_size[0] - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0.0, image_size[1] - 1)

        # normalize
        xy /= image_size

        # normalize position (x, y) respectively
        pos_df_norm = pd.DataFrame({
            "frame": pos_df["frame"],
            "id": pos_df["id"],
            "x": xy[:, 0],
            "y": xy[:, 1]
        })
        return pos_df_norm

    @staticmethod
    def thin_out_pos_df(pos_df, interval):
        all_frames = pos_df["frame"].unique()
        remained_frames = np.arange(all_frames[0], all_frames[-1] + 1, interval)
        remained_rows = pos_df["frame"].isin(remained_frames)

        pos_df_thinned_out = pos_df[remained_rows]
        pos_df_thinned_out = pos_df_thinned_out.reset_index(drop=True)

        return pos_df_thinned_out
