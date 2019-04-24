import json
import os
from argparse import Namespace, ArgumentParser

import matplotlib
import platform
if (platform.system() != "Windows"):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from data_utils import obs_pred_split
from evaluation_metrics import compute_abe, compute_fde
from load_model_config import load_model_config
from my_social_model import MySocialModel
from provide_train_test import provide_train_test
from vizualize_trajectories import visualize_trajectories

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def _load_eval_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--trained_model_config", type=str, default="data/results/20181123121153_circle/test=zara1/zara1.json")
    parser.add_argument("--trained_model_file", type=str, default="data/results/20181123121153_circle/test=zara1/social_train_model_e0008.h5")
    return parser.parse_args()


def main() -> None:
    # 'circle', 'rect
    grid_type = 'rect'
    is_test = True

    args = _load_eval_args()
    config = load_model_config(args.trained_model_config)
    config.grid_type = grid_type
    config.test_dataset_kind = 'students003'

    out_dir = os.path.join(os.path.dirname(args.trained_model_file), "eval")
    os.makedirs(out_dir, exist_ok=True)

    # load data
    _, test_data = provide_train_test(config, is_test=is_test)

    obs_len_test, pred_len_test = obs_pred_split(
        config.obs_len, config.pred_len, *test_data)

    # load trained model weights
    my_model = MySocialModel(config)
    my_model.train_model.load_weights(args.trained_model_file)

    # first, predict `pred_len` sequences following observation sequence
    x_obs_len_test, _, grid_obs_len_test, zeros_obs_len_test = obs_len_test
    x_pred_len_test, *_ = pred_len_test

    x_pred_len_model = my_model.sample_model.predict(
        [x_obs_len_test, grid_obs_len_test, zeros_obs_len_test],
        batch_size=config.batch_size, verbose=1)

    # --------------------------------------------------------------------------
    # visualization
    # --------------------------------------------------------------------------

    x_concat_test = np.concatenate([x_obs_len_test, x_pred_len_test], axis=1)
    x_concat_model = np.concatenate([x_obs_len_test, x_pred_len_model], axis=1)

    # visualize true and predicted trajectories, and save as png files
    out_fig_dir = os.path.join(out_dir, "figs")
    os.makedirs(out_fig_dir, exist_ok=True)
    for s in range(len(x_concat_test)):
        fig = visualize_trajectories(x_concat_test[s], x_concat_model[s],
                                     config.obs_len, config.pred_len)
        fig_file = os.path.join(out_fig_dir, "{0:04d}.png".format(s))
        fig.savefig(fig_file)
        plt.close(fig)

    # --------------------------------------------------------------------------
    # evaluation
    # --------------------------------------------------------------------------

    # evaluation
    ade = compute_abe(x_pred_len_test, x_pred_len_model)
    fde = compute_fde(x_pred_len_test, x_pred_len_model)
    report = {"ade": float(ade), "fde": float(fde)}

    # write to a json file
    report_file = os.path.join(out_dir, "report.json")
    with open(report_file, "w") as f:
        json.dump(report, f)

    print("Average displacement error: {}".format(ade))
    print("Final displacement error: {}".format(fde))


if __name__ == '__main__':
    main()
