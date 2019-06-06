#加载所使用到的库
import os
from argparse import Namespace, ArgumentParser
from shutil import copyfile

import matplotlib
import platform
if (platform.system() != "Windows"):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_utils import obs_pred_split
from general_utils import dump_json_file, load_json_file
from general_utils import now_to_str
from load_model_config import ModelConfig
from load_model_config import load_model_config
from my_social_model import MySocialModel
from provide_train_test import provide_train_test

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_train_args() -> Namespace:  #默认输入输出路径
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="data/configs/zara1.json")#配置文件所在的路径
    parser.add_argument("--out_root", type=str, default="data/results")#将训练结果存至该路径下
    return parser.parse_args()


def _make_weights_file_name(n_epochs: int) -> str:   #返回训练权重
    return "social_train_model_e{0:04d}.h5".format(n_epochs)

def plt_save_loss(config, out_dir, history_obj):   #保存训练过程loss曲线图

#绘制训练过程中的loss曲线
    plt.plot(history_obj["loss"])
    plt.plot(history_obj["val_loss"])
    plt.title("social model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper right")
    plt.savefig(os.path.join(out_dir, "test={}_loss.png".format(
        config.test_dataset_kind)))

def train_generator(input_list, batch_size):

    start = 0  # 开始迭代的起始位置

    x_obs_len_train, grid_obs_len_train, zeros_obs_len_train, y_pred_len_train = input_list
    while True:
        stop = start + batch_size
        if stop >  x_obs_len_train.shape[0] -1:
            start = x_obs_len_train.shape[0] -1 - batch_size
        each_obs = x_obs_len_train[start: stop]
        each_grid = grid_obs_len_train[start: stop]
        each_zeros = zeros_obs_len_train[start: stop]

        y_train = y_pred_len_train[start: stop]
        start += batch_size

        yield [each_obs, each_grid, each_zeros], y_train

def train_social_model(out_dir: str, config: ModelConfig) -> None:
    # 加载数据
    train_data, test_data = provide_train_test(config)

    # 训练数据
    obs_len_train, pred_len_train = obs_pred_split(config.obs_len,
                                                   config.pred_len,
                                                   *train_data)
    x_obs_len_train, _, grid_obs_len_train, zeros_obs_len_train = obs_len_train
    _, y_pred_len_train, _, _ = pred_len_train

    # 测试数据
    obs_len_test, pred_len_test = obs_pred_split(config.obs_len,
                                                 config.pred_len,
                                                 *test_data)
    x_obs_len_test, _, grid_obs_len_test, zeros_obs_len_test = obs_len_test
    _, y_pred_len_test, _, _ = pred_len_test

    os.makedirs(out_dir, exist_ok=True)

    # 训练部分，加载历史运动轨迹，batch_size,训练批次
    my_model = MySocialModel(config)
    print(config.steps_per_epoch)

    history = my_model.train_model.fit(
        [x_obs_len_train, grid_obs_len_train, zeros_obs_len_train],
        y_pred_len_train,
        batch_size=config.batch_size,
        epochs=config.n_epochs,
        verbose=1,
       
        validation_data=(
            [x_obs_len_test, grid_obs_len_test, zeros_obs_len_test],
            y_pred_len_test
        )
    )

    # 保存训练权重
    weights_file = os.path.join(out_dir,
                                _make_weights_file_name(config.n_epochs))
    my_model.train_model.save_weights(weights_file)

    history_file = os.path.join(out_dir, "history.json")
    dump_json_file(history.history, history_file)

    # 保存训练过程loss曲线图
    plt_save_loss(config, out_dir, history.history)


def main():
    is_plt_loss = False

    # 'circle', 'rect
    grid_type = 'circle'    #设置训练的模型是矩形邻域还是圆形邻域

    args = load_train_args()
    config = load_model_config(args.config)
    config.data_root = os.path.abspath(config.data_root)
    now_str = now_to_str() + '_' + grid_type
    config.grid_type = grid_type

    out_dir = os.path.join(args.out_root, "{}".format(now_str),
                           "test={}".format(config.test_dataset_kind))

    if is_plt_loss:
        my_model = MySocialModel(config)
        weights_file = 'data/results/20181122135218_circle/test=zara1/social_train_model_e0002.h5'  
        my_model.train_model.save_weights(weights_file)

        history_file = os.path.join(out_dir, "history.json")
        history_obj = load_json_file(history_file)
        plt_save_loss(config, out_dir, history_obj)
    else:
        train_social_model(out_dir, config)

    copyfile(args.config, os.path.join(out_dir, os.path.basename(args.config)))


if __name__ == '__main__':
    main()
