#coding:utf-8
from glob import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import string, random

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def plt_show_predict(avi_path, view_path,i):
    view_dir = os.path.join(view_path,'file/view.txt')
    # 先读取预测结果文件
    #file_obj = open(file_path_dir).readlines()
    #print(file_obj)
    key_name = 0
    
    #读取坐标信息    
    with open(view_dir,'r') as vf:
        view_infor = vf.readlines()
    #print(view_infor) #['11.mp4:person_a,633 365 353 411:person_b,325 403 41 449:90\n']
    person={}
    
    infor = view_infor[i].strip().split(':')
    video_name = infor[0]
    frame_num = int(infor[-1])
    person_num = len(infor)-2
    #print(infor,video_name,frame_num,person_num)
    for j in range(person_num):
        person[j] = infor[j+1]
    #print(person)    

    # 空字典用于存储预测结果
    coor_dic = dict()
    # 循环读取每一行预测结果记录
    x_0,y_0,x_n,y_n = person[0].split()
    x_0 = int(x_0)
    x_n = int(x_n)
    y_0 = int(y_0)
    y_n = int(y_n)

    #print(x_0,x_n,y_0,y_n)
    
    '''
    for ind, each_line in enumerate(file_obj):
        img_name, coor = each_line.split(':')
        if ind == 0:
            key_name = int(img_name.split('_')[-1])
            coor_dic[key_name] = []
        x, y = coor.split(',')
        # 每一帧预测结果保存到字典中
        coor_dic[key_name].append([int(x), int(y)])
    #print(coor_dic)
    '''
    # 视频读取
    video_path = os.path.join(avi_path +'/'+video_name)
    print(video_path)
    cap = cv2.VideoCapture(video_path)
    print(cap.isOpened())

    #frames_num=cap.get(7)
    coor_x = [int(x_0+i*(x_n-x_0)/frame_num) for i in range(int(frame_num))]
    coor_y = [int(y_0+i*(y_n-y_0)/frame_num) for i in range(int(frame_num))]
    #print(len(coor_y),len(coor_x))
    # 获取预测的x、y坐标信息
    #begin_frame = list(coor_dic.keys())[0]
    #print(begin_frame)
    #coor_x = np.array(coor_dic[begin_frame])[:, 0]
    #coor_y = np.array(coor_dic[begin_frame])[:, 1]
    #print(len(coor_x))
    #print(coor_y)
    show_id = 0
    # 循环读取每一帧，根据预测结果，在每一帧上标记出预测的行人轨迹信息
    while (cap.isOpened()):
        frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
        #print(frame_id)
        if frame_id > frame_num :
            break
        ret, img = cap.read()
        #print('bf is {}, fi is {}'.format(begin_frame,frame_id)) 
        # 由于预测的不是视频的第一帧，所以需要过滤掉没有预测过的那些帧
        #if begin_frame > frame_id:
        #    continue
            #pass
        #print(show_id)
        # 每10帧显示一次
        if show_id + 10  <= len(coor_x):
            pos_x = coor_x[show_id]
            pos_y = coor_y[show_id]
            #print(pos_x,pos_y)
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
            #if show_id + (x_n-x_0)/5 >frame_num or show_id + (x_n-x_0)/5 <0:
            #    break
            pre_x = np.linspace(pos_x, np.mean(coor_x[show_id] + (x_n-x_0)/3), 5) + np.random.rand(5) * 11
            #pirint(pos_x,np.mean(coor_x[show_id +10]))
            pre_x = pre_x.astype(np.uint16)
            #print('pre_x is {}'.format(pre_x))
            pre_y = np.linspace(pos_y, np.mean(coor_y[show_id] + (y_n-y_0)/3), 5) + np.random.rand(5) * 11
            pre_y = pre_y.astype(np.uint16)
            #print('pre_y is {}'.format(pre_y))
            pos_x_pre =  pos_x
            pos_y_pre =  pos_y
   
            for ind in range(1, 5, 1):
                cv2.line(img, (pos_x, pos_y), (pre_x[ind], pre_y[ind]), (255, 0, 0), 2)
                cv2.circle(img,(pre_x[ind],pre_y[ind]),2,(255, 0 ,0),2)
                pos_x = pre_x[ind]
                pos_y = pre_y[ind]
         
                         
            
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
    avi_path = '/home/leonard/skk/yolov3-track/预测/'
    view_path = '/home/leonard/skk/yolov3-track/预测/'
    # 0:11.mp4, 1: 
    plt_show_predict(avi_path,view_path,1)
    

