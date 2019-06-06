# Requirements 运行环境

- Python 3.6.0+
- Keras 2.1.5+
- TensorFlow 1.6.0+
- Opencv3.4 + contrib
***
# Pedestrian detection 行人检测
-- yolo行人检测主程序
```yolo.py ```

-- 执行程序
> run ```yolo_video.py ``` 

## Preparation
[voc数据集](http://host.robots.ox.ac.uk/pascal/VOC/)

按论文需求修改配置文件yolo.cfg、voc_annotation.py、voc_class.txt

##  Training

运行下面语句训练检测模型
> run ```python3 train.py```

训练好的权重放在model_data路径下

## Testing
行人检测跟踪的测试，运行下面语句，input是输入视频，output是保存路径
 
> run ```python yolo_video.py --input test.mp4 --output cuc1.mp4```

***
# Pedestrian prediction 行人预测
## Preparation 
先设置config里面的配置文件
Set dataset attribute of the config files in `configs/.`

## Training
训练行人预测网络，运行下面语句，config设置网络超参数，out root路径保存模型

>run ```python3 train_social_model.py --config ./data/configs/ucy.jason --out root ./data/result/20190527/test=ucy/social_train_model_e0010.h5```

## Testing
测试行人预测网络，运行下面语句，
Run evaluate_social_model.py， trained_model_config设置测试的超参数，trained_model_file路径指定测试使用的模型
> run ```python3 evaluate_social_my_model.py  --trained_model_config  /home/leonard/skk/social_lstm_keras_tf-master/data/configs/other.json --trained_model_file   /home/leonard/skk/social_lstm_keras_tf-master/data/results/20190527/test=ucy/social_train_model_e0010.h5```

## 融合模型执行文件
测试融合模型，运行下面语句，input是待测试视频
Run human_track_predict.py 
> run ```python3.5 human_track_predict.py --input test.mp4 ```


## 论文模型和原来模型的对比
（由于时间原因，只标注了一个视频，也是在校园里采集的，由于角度问题，行人的移动变化明显，所以便于可视化）

下图对比了改进后的圆形邻域预测模型与原来的矩形邻域预测模型，
- 红色曲线是真实值;
- 蓝色曲线是改进后的模型;
- 绿色曲线是原来的矩形邻域模型。


![(1)](https://github.com/YapingZ/Pedestrian-behavior-trajectory-prediction/blob/master/picture/2.png)   ![(2)](https://github.com/YapingZ/Pedestrian-behavior-trajectory-prediction/blob/master/picture/0.png)


![(3)](https://github.com/YapingZ/Pedestrian-behavior-trajectory-prediction/blob/master/picture/4.png)   ![(4)](https://github.com/YapingZ/Pedestrian-behavior-trajectory-prediction/blob/master/picture/3.png)

## 所有的程序合成一个文件human_track_predict.py

输入一个待测视频，首先经过检测跟踪模型，调用yolo.py文件里的detect_img和detect_video函数进行行人的检测跟踪，将坐标保存至person.vsp；然后调用evaluate_social_model.py文件进行预测模型的测试，将行人的预测坐标保存至txt文件；最后调用可视化文件read_and_show输出预测视频。

以下是模型的整个测试流程：


![执行流图](https://github.com/YapingZ/Pedestrian-behavior-trajectory-prediction/blob/master/picture/1.png)

## Restrictions
- work only on batch size = 1
- require much RAM (use almost all 16GB in my environment)
