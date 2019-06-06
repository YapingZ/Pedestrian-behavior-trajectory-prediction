import os

import numpy as np
import pandas as pd

from general_utils import get_image_size

#数据预处理器
class UcyDatasetPreprocessor:
    ped_start_line_words = "Num of control points"
    vsp_columns = ["x", "y", "frame", "gaze", "dummy1", "dummy2", "dummy3",
                   "dummy4"]

    # 设置帧间隔，隔10帧取一帧
    frame_interval = 10

    def __init__(self, data_dir, dataset_kind):
        self._data_dir = data_dir
        self._dataset_kind = dataset_kind

    def preprocess_frame_data(self):
        lines = self._read_lines(self._get_vsp_file(self._data_dir))

        # 找到指定每个行人轨迹的第一行
        ped_start_indices = [li for li, line in enumerate(lines) if
                             line.find(self.ped_start_line_words) != -1]

        pos_df_raw = []

        # 提取行人位置，在UCY数据集中，没有给出行人id，因此为行人添加相应的id
        for i, start_index in enumerate(ped_start_indices):
            n_pos_i = int(lines[start_index].split()[0])
            pos_lines_i = lines[start_index + 1:start_index + 1 + n_pos_i]
            pos_df_raw_i = pd.DataFrame([line.split() for line in pos_lines_i],
                                        columns=self.vsp_columns)
           
            pos_df_raw_i["id"] = i + 1

            pos_df_raw.append(pos_df_raw_i)

        pos_df_raw = pd.concat(pos_df_raw)
        # 仅保留（frame，id，x，y），并不读取角度信息
        pos_df_raw = pos_df_raw[["frame", "id", "x", "y"]].astype(np.float32)
        pos_df_raw = pos_df_raw.reset_index(drop=True)

        # interpolate，插值 ； normalize，标准化
        pos_df_preprocessed = self.interpolate_pos_df(pos_df_raw)
        pos_df_preprocessed = self.normalize_pos_df(pos_df_preprocessed,
                                                    self._dataset_kind)
        pos_df_preprocessed = self.thin_out_pos_df(pos_df_preprocessed,
                                                   self.frame_interval)
        pos_df_preprocessed = pos_df_preprocessed.sort_values(["frame", "id"])

        return pos_df_preprocessed

        #加载坐标文件
    def _get_vsp_file(data_dir):
        vsp_file_name = "{}.vsp".format(os.path.basename(data_dir))
        vsp_file = os.path.join(data_dir, vsp_file_name)

        return vsp_file


    def _read_lines(file):
        with open(file, "r") as f:
            return f.readlines()

        
		
         #从scipy库中导入插值函数interpolate
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


    def normalize_pos_df(pos_df, dataset_kind):
        image_size = np.array(get_image_size(dataset_kind))

        xy = np.array(pos_df[["x", "y"]])
        
        xy += image_size / 2
        # 裁剪
        xy[:, 0] = np.clip(xy[:, 0], 0.0, image_size[0] - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0.0, image_size[1] - 1)

        # 标准化
        xy /= image_size

        # 分别标准化位置（x，y）
        pos_df_norm = pd.DataFrame({
            "frame": pos_df["frame"],
            "id": pos_df["id"],
            "x": xy[:, 0],
            "y": xy[:, 1]
        })
        return pos_df_norm


    def thin_out_pos_df(pos_df, interval):
        all_frames = pos_df["frame"].unique()
        remained_frames = np.arange(all_frames[0], all_frames[-1] + 1, interval)
        remained_rows = pos_df["frame"].isin(remained_frames)

        pos_df_thinned_out = pos_df[remained_rows]
        pos_df_thinned_out = pos_df_thinned_out.reset_index(drop=True)

        return pos_df_thinned_out
