# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer
import math
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
import cv2
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model
tracknum=[]
allpeople=0
savefile=open('person.vsp','w')
framenum=0

#加载训练好的模型
class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score" : 0.4,
        "iou" : 0.5,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        '''
        读取类别名
        '''
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        '''
        取anchors比率
        '''
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes
#把行人框出来
    def detect_image(self, image,num):
        start = timer()
        humanplace=[]
        # 保证图片的尺寸是32的倍数
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size))) # resize image with unchanged aspect ratio using padding
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        #print(image_data.shape)
        # 归一化图像像素
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        #print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        # 加载字体
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300 # 2
        tempframe=str(num)+'\n' #
        

        # 画框
        hunamnum=0
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            dian_y=int((top+bottom)/2)
            dian_x=int((left+right)/2)
            temppoint=[dian_x,dian_y]

            #print(label)
            if 'person' in label:
                temp=label+' ('+str(left)+','+str(top)+')  ('+str(right)+','+str(bottom)+')'+'\n'

                humanplace.append([left,top,right,bottom])
                hunamnum=hunamnum+1
                tracknum.append(temppoint)
            #print(label, (left, top), (right, bottom))


            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            if 'person' in label:
                # My kingdom for a good redistributable image drawing library.
                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[c])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=self.colors[c])
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                del draw
        pointdraw = ImageDraw.Draw(image)
        for i in tracknum:
            print(len(tracknum))
            pointdraw.ellipse((i[0] - 5, i[1] - 5, i[0] + 5, i[1] + 5), (255, 0, 0))
        del pointdraw
        end = timer()
        #print(end - start)
        return image,humanplace

    def close_session(self):
        self.sess.close()


def writedown(temppoint,humannum,recogframe,video_size):
    for k in range(0,humannum):
        # pointlist_x=[]
        # pointlist_y=[]
        # pointframe=[]
        # pointangle=[]
        temp=str(int((recogframe-2)/7))
        temp=temp.rstrip('.0')
        savefile.write(temp+' - Num of control points\n')#表示有几个点
        for m in range(7,recogframe-1,7):#这边的7表示间隔几帧计算
            temp_x=int(temppoint[k][m][0]-video_size[0]/2)#目前位置x
            tempago_x=int(temppoint[k][m-7][0]-video_size[0]/2)#前7帧位置x
            temp_y=int(temppoint[k][m][1]-video_size[1]/2)#目前位置y
            tempago_y = int(temppoint[k][m-7][1] - video_size[1] / 2)#前7帧位置y
            #计算角度
            if temp_x<tempago_x and temp_y>tempago_y:
                tempangle=-math.atan((temp_x-tempago_x)/(temp_y-tempago_y))*180/math.pi-180

            if temp_x<tempago_x and temp_y<tempago_y:
                tempangle = -math.atan((temp_x - tempago_x)/ (temp_y - tempago_y)) * 180 / math.pi

            if temp_x>tempago_x and temp_y>tempago_y:
                tempangle=180-math.atan((temp_x - tempago_x)/(temp_y - tempago_y)) * 180 / math.pi

            if temp_x > tempago_x and temp_y < tempago_y:
                tempangle = math.atan((temp_x - tempago_x)/(temp_y - tempago_y)) * 180 / math.pi

            if temp_x > tempago_x and temp_y == tempago_y:
                tempangle=90
            if temp_x < tempago_x and temp_y == tempago_y:
                tempangle=-90
            if temp_y==tempago_y and tempago_x==temp_x:
                tempangle=0
            if tempago_x == temp_x and temp_y < tempago_y:
                tempangle = 0
            if tempago_x == temp_x and temp_y > tempago_y:
                tempangle = 180
            tempstr=str(temp_x)+' '+str(temp_y)+' '+str(temppoint[k][m][2])+' '+str(tempangle)+' - (2D point, m_id)\n'
            savefile.write(tempstr)
            #保存行人的跟踪坐标








def jugyment(temppoint, humannum,recogframe):
    for i in range(0,humannum):
        tempnewplace=[0,0]
        tempoldplace=[0,0]
        for k in range(recogframe):
            if temppoint[i][k][0]==0:
                tempnewplace[0]=temppoint[i][k-1][0]
                tempnewplace[1]=temppoint[i][k-1][1]
                tempoldplace[0]=temppoint[i][k-6][0]
                tempoldplace[1]=temppoint[i][k-6][1]
                break
        placelength=math.pow(tempnewplace[0]-tempoldplace[0],2)+math.pow(tempnewplace[1]-tempoldplace[1],2)
        if placelength<=4:
            return 0
    return 1







#
#
#
#先视频分帧
#调用上面的detect_image函数，把人框出来
#用封装好的卡尔慢跟踪器根据相邻帧的相似度关系，把相似度高的归为同一人，进行跟踪

def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path) #读取视频
    
    
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    =  cv2.VideoWriter_fourcc(*'MPEG')  #保存视频格式mp4
    video_fps       = vid.get(cv2.CAP_PROP_FPS)  #视频帧率
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),  #图片大小
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter('cuc1.mp4', video_FourCC, 24, video_size)  #以24秒/帧输出
    #isOutput = True if output_path != "" else False
    #if isOutput:
        #print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))

    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()

    recogframe=100#保存多少帧的变量
    p=0  #视频帧的启示
    k=0
    init_once = False
    humantrack=[]#人的位置
    huamanplace=[]
    global framenum#保存帧数
    global allpeople#保存人数
    trackers = cv2.MultiTracker_create()#建立多目标跟踪器
    temppoint=np.zeros((100,recogframe,3))#创建保存人的位置的空间
    while True:
        return_value, frame = vid.read()
        if return_value:
            framenum = framenum + 1
        else:
            print('我跑完了')
            break

        if k==0:#进入yolo检测人
            init_once = False
            image = Image.fromarray(frame)
            image,huamanplace = yolo.detect_image(image,p)#调用上边detect_image函数，把人框出来，返回图片和获取人的位置
            p=p+1
            result = np.asarray(image)
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(255, 0, 0), thickness=2)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            #out.write(result)
            cv2.imshow("result", result)

        print(huamanplace)
        peoplenum=len(huamanplace)#人数

        k = k + 1
        if peoplenum==0:#如果没有检测到人至k=0 重新检测
            k=0
        if not init_once:#初始化追踪器
            allpeople=allpeople+peoplenum
            for i in huamanplace:
                temp=(i[0],i[1],i[2]-i[0],i[3]-i[1])
                #print(temp)
                ok=trackers.add(cv2.TrackerMedianFlow_create(),frame,temp)#在多目标追踪器中添加单目标追踪
            for q in range(0,peoplenum):#添加人的初始位置
                temppoint[q][k-1]=[int((huamanplace[q][0]+huamanplace[q][2])/2),int((huamanplace[q][3]+huamanplace[q][1])/2),framenum]

            init_once=True

        ok,boxes=trackers.update(frame)#更新追踪器 返回人的位置
      

        boxnum=0
        for newbox in boxes:#boxes为返回人的位置

            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(frame, p1, p2, (200, 0, 0))
            tempplace=(int(newbox[0])+int(newbox[2]/2),int(newbox[1])+int(newbox[3]/2))
            temppoint[boxnum][k-1]=[int(newbox[0])+int(newbox[2]/2),int(newbox[1])+int(newbox[3]/2),framenum]#计算人x,y和帧数
            boxnum=boxnum+1
            humantrack.append(tempplace)
            for po in humantrack:
                cv2.circle(frame,po,2,(0,255,0))
        out.write(frame)
        cv2.imshow('1',frame)
        cv2.waitKey(20)
        if k==recogframe:#当到达100帧后写入一次人轨迹，k至0重新检测人
            k=0
            trackers = cv2.MultiTracker_create()
            writedown(temppoint, peoplenum, recogframe,video_size)
            temppoint = np.zeros((100, recogframe, 3))
        if k>10:
            ifstop=jugyment(temppoint,peoplenum,recogframe)
            if ifstop==0:
                trackers=cv2.MultiTracker_create()
                writedown(temppoint,peoplenum,k-1,video_size)
                temppoint=np.zeros((100,recogframe,3))
                k = 0

        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    yolo.close_session()

