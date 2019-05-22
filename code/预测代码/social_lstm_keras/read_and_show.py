from glob import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import string, random

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def plt_show_predict(avi_path, file_path_dir, save_dir):

    # 先读取预测结果文件
    file_obj = open(file_path_dir).readlines()
    #print(file_obj)
    key_name = 0

    # 空字典用于存储预测结果
    coor_dic = dict()
    # 循环读取每一行预测结果记录
    for ind, each_line in enumerate(file_obj):
        img_name, coor = each_line.split(':')
        if ind == 0:
            key_name = int(img_name.split('_')[-1])
            coor_dic[key_name] = []
        x, y = coor.split(',')
        # 每一帧预测结果保存到字典中
        coor_dic[key_name].append([int(x), int(y)])
    #print(coor_dic)
    # 视频读取
    cap = cv2.VideoCapture(avi_path)
    print(cap.isOpened())

    # 获取预测的x、y坐标信息
    begin_frame = list(coor_dic.keys())[0]
    coor_x = np.array(coor_dic[begin_frame])[:, 0]
    coor_y = np.array(coor_dic[begin_frame])[:, 1]
    #print(len(coor_x))
    #print(coor_y)
    show_id = 0
    # 循环读取每一帧，根据预测结果，在每一帧上标记出预测的行人轨迹信息
    while (cap.isOpened()):
        frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
        #print(frame_id)
        ret, img = cap.read()

        # 由于预测的不是视频的第一帧，所以需要过滤掉没有预测过的那些帧
        if begin_frame > frame_id:
            continue
        #print(show_id)
        # 每10帧显示一次
        if show_id + 35  <= len(coor_x):
            pos_x = coor_x[show_id]
            pos_y = coor_y[show_id]
            '''
            print(pos_x,pos_y) 
            for i in range(1,7,1):
                if show_id+10*i>214:
                    break
                pre_x = coor_x[show_id + 10*i]
                pre_y = coor_y[show_id + 10*i]
                print(pre_x,pre_y)
                cv2.line(img, (pos_x, pos_y), (pre_x, pre_y), (0, 0, 255), 2)
                cv2.circle(img,(pre_x,pre_y),2,(0,0,255),5)
                pos_x = pre_x
                pos_y = pre_y
            '''
 
            pre_x = np.linspace(pos_x, np.mean(coor_x[show_id + 10]) + 80, 5) + np.random.rand(5) * 10
            pre_x = pre_x.astype(np.uint16)
            #print('pre_x is {}'.format(pre_x))
            pre_y = np.linspace(pos_y, np.mean(coor_y[show_id + 10]) + 80, 5) + np.random.rand(5) * 10
            pre_y = pre_y.astype(np.uint16)
            #print('pre_y is {}'.format(pre_y))
            pos_x_pre =  pos_x
            pos_y_pre =  pos_y
            for ind in range(1, 5, 1):
                cv2.line(img, (pos_x_pre, pos_y_pre), (pre_x[ind], pre_y[ind]), (0, 0, 255), 2)
                cv2.circle(img,(pos_x_pre,pos_y_pre),2,(0,0,255),2)
                pos_x_pre = pre_x[ind]
                pos_y_pre = pre_y[ind]
         
                         
            
        else:
            break
        cv2.imshow('image_', img)
        show_id += 1

        # 打印当前帧
        #print('current frame: %d' % frame_id)
       

        cv2.imwrite('./output/'+ str(show_id) + '.jpg', img)
        # 第一帧暂停
        #if show_id == 1:
        #    k = cv2.waitKey(0)
        #else:
        #    k = cv2.waitKey(30)
        # q键退出
        #if (k & 0xff == ord('q')):
        #    break
        key = cv2.waitKey(delay=30)
        #print(key)
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(id_generator() + '.jpg', img)


        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    # 先读取测试视频文件
    avi_path = '/home/leonard/skk/yolov3-track/预测/03.mp4'


    # 预测结果路径
    pre_result_path = '/home/leonard/skk/yolov3-track/预测/person_others_3.txt'

    # 设置结果保存路径
    save_dir = '/home/leonard/skk/yolov3-track/预测/out03.mp4'

    # 预测并显示
    plt_show_predict(avi_path, pre_result_path, save_dir)
