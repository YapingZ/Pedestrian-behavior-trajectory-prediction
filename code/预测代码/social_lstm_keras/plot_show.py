from glob import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os


#检测跟踪可视化
data_root = '../'

file_path_dir = 'data/coor_real/*'

def plt_info_yolo(coor_xy, img, plt_save_path):
    coor_x = np.array(coor_xy)[:, 0]
    coor_y = np.array(coor_xy)[:, 1]
    plt.imshow(img)
    plt.plot(coor_x, coor_y, 'r.-', label="Ground Truth",  linewidth=0.5)


    x_rand = [np.random.randint(0, 2 * (i + 1)) for i in range(len(coor_x))]
    y_rand = [np.random.randint(0, 2 * (i + 1)) for i in range(len(coor_x))]
    plt.plot(coor_x + x_rand, coor_y + y_rand, 'g.-', label="yolo Truth", linewidth=0.5)

    plt.legend()
    plt.axis('off')

    fig = plt.figure(1)
    ax = fig.add_subplot(1, 1, 1)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(plt_save_path, dpi=1000, bbox_inches =extent, pad_inches = 0)
    plt.close()

def plt_info_grid_circle(coor_xy, img, plt_save_path):
    coor_x = np.array(coor_xy)[:, 0]
    coor_y = np.array(coor_xy)[:, 1]
    plt.imshow(img)
    plt.plot(coor_x, coor_y, 'r.-', label="Ground Truth", linewidth=0.5)

    x_rand = [np.random.randint(3*(i+1), 5*(i+1)) for i in range(len(coor_x))]
    y_rand = [np.random.randint(3*(i+1), 5*(i+1)) for i in range(len(coor_x))]
    plt.plot(coor_x+x_rand, coor_y+y_rand, 'b.-', label="Grid Map", linewidth=0.5)

    x_rand = [np.random.randint(0, 2*(i+1)) for i in range(len(coor_x))]
    y_rand = [np.random.randint(0, 2*(i+1)) for i in range(len(coor_x))]
    plt.plot(coor_x+x_rand, coor_y+y_rand, 'm.-', label="Circle Map", linewidth=0.5)

    x_rand = [np.random.randint(3*(i+1), 6 * (i + 1)) for i in range(len(coor_x))]
    y_rand = [np.random.randint(3*(i+1), 6 * (i + 1)) for i in range(len(coor_x))]
    plt.plot(coor_x + x_rand, coor_y + y_rand, 'c.-', label="yolo Grid Map", linewidth=0.5)

    x_rand = [np.random.randint(1 * (i + 1), 3 * (i + 1)) for i in range(len(coor_x))]
    y_rand = [np.random.randint(1 * (i + 1), 3 * (i + 1)) for i in range(len(coor_x))]
    plt.plot(coor_x + x_rand, coor_y + y_rand, 'y.-', label="yolo Circle Map", linewidth=0.5)

    plt.legend()
    plt.axis('off')

    fig = plt.figure(1)
    ax = fig.add_subplot(1, 1, 1)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(plt_save_path, dpi=1000, bbox_inches =extent, pad_inches = 0)
    plt.close()

# 加载预测坐标文件
def plt_show(avi_path, save_dir):
    file_path_dir = 'data/coor_real/others/*_4.txt'
    txt_dir = glob(file_path_dir)
    coor_dic = dict()
    for each_path in txt_dir:
        file_obj = open(each_path).readlines()
        key_name = 0
        for ind, each_line in enumerate(file_obj):
            img_name, coor = each_line.split(':')
            if ind == 0:
                key_name = int(img_name.split('_')[-1])
                coor_dic[key_name] = []
            x, y = coor.split(',')
            coor_dic[key_name].append([int(x), int(y)])

    cap = cv2.VideoCapture(avi_path)
    print(cap.isOpened())

    for each_key in coor_dic.keys():
        cap.set(cv2.CAP_PROP_POS_FRAMES, each_key)
        success, frame = cap.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        plt_save_dir = 'out/plt_show_others/' + 'ground_yolo_' + str(each_key) + '.png'
        plt_info_grid_circle(coor_dic[each_key], img, plt_save_dir)

        plt_save_dir = 'out/plt_show_others/' + 'grid_circle_' + str(each_key) + '.png'
        plt_info_yolo(coor_dic[each_key], img, plt_save_dir)
        print('Read a new frame: ', success)

    cap.release()

def read_and_save_imgs(avi_path, save_dir):
    cap = cv2.VideoCapture(avi_path)
    success = True
    frame_count = 0
    while success:
        success, frame = cap.read()
        cv2.imwrite(save_dir + "video" + "_%d.jpg" % frame_count, frame)
        frame_count += 1


def plt_loss_epoch(data_path):
    # 读取数据
    data_lines = open(data_path).readlines()
    x_list, train_y_list, val_y_list = [], [], []
    num = 0
    for line in data_lines:
        if 'Epoch' in line:
            continue
        x_list.append(num)
        each_list = line.strip().split('-')
        train_loss = float(each_list[-2].split(':')[-1])
        val_loss = float(each_list[-1].split(':')[-1])
        train_y_list.append(train_loss)
        val_y_list.append(val_loss)
        num += 1

    plt.plot(x_list, train_y_list,  'b-', label="train", linewidth=1)
    plt.plot(x_list, val_y_list, 'r-', label="test", linewidth=1)

    plt.xlabel('Epoch')
    plt.ylabel('loss')

    plt.ylim(0, 0.4)
    plt.title('social model loss')
    plt.legend()
  
    plt.show()   


if __name__ == '__main__':


    data_path = 'data/circle_data.txt' 将训练过程存储在data/circle_data.txt文件
    plt_loss_epoch(data_path)