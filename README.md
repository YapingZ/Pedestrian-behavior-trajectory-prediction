# Requirements

- Python 3.6.0+
- Keras 2.1.5+
- TensorFlow 1.6.0+
- Opencv3.4 + contrib
***
# Pedestrian detection
-- 主程序
```yolo.py ```

-- 执行程序
> run ```yolo_video.py ``` 

## Preparation
[voc数据集](http://host.robots.ox.ac.uk/pascal/VOC/)

按论文需求修改配置文件yolo.cfg、voc_annotation.py、voc_class.txt

##  Training
Modify train.py and start training
run ```python3 train.py```

## Replacement weight
将训练好的权重放在model_data路径下

## Testing
Run yolo_video.py
> ```python yolo_video.py --input test.mp4 --output test1.mp4```

***
# Pedestrian prediction
## Preparation
Set dataset attribute of the config files in `configs/.`

## Training
Run train_social_model.py.
> ```python3 train_social_model.py --config ./data/configs/ucy.jason --out root ./data/result/20190527/test=ucy/social_train_model_e0010.h5```

## Testing
Run evaluate_social_model.py
> ```python3 evaluate_social_my_model.py  --trained_model_config  /home/leonard/skk/social_lstm_keras_tf-master/data/configs/other.json --trained_model_file   /home/leonard/skk/social_lstm_keras_tf-master/data/results/20190527/test=ucy/social_train_model_e0010.h5```

## 融合模型执行文件
Run human_track_predict.py
> ```python3.5 human_track_predict.py --input test.mp4 ```

## Restrictions
- work only on batch size = 1
- require much RAM (use almost all 16GB in my environment)


## 论文模型和原来模型的对比

下图对比了改进后的圆形邻域预测模型与原来的矩形邻域预测模型，红色曲线是真实值（由于时间原因，只标注了一个视频，也是在校园里采集的，由于角度问题，行人的移动变化明显，所以便于可视化）;蓝色曲线是改进后的模型;绿色曲线是原来的矩形邻域模型。
![(1)](https://github.com/YapingZ/Pedestrian-behavior-trajectory-prediction/blob/master/picture/2.png)  ![(2)](https://github.com/YapingZ/Pedestrian-behavior-trajectory-prediction/blob/master/picture/3.png)
![(3)](https://github.com/YapingZ/Pedestrian-behavior-trajectory-prediction/blob/master/picture/3.png)  ![(4)](https://github.com/YapingZ/Pedestrian-behavior-trajectory-prediction/blob/master/picture/4.png)

## 所有的程序合成一个文件human_track_predict.py

输入一个待测视频，首先经过检测跟踪模型，调用yolo.py文件里的detect_img和detect_video函数进行行人的检测跟踪，将坐标保存至person.vsp；然后调用evaluate_social_model.py文件进行预测模型的测试，将行人的预测坐标保存至txt文件；最后调用可视化文件read_and_show输出预测视频。

以下是模型的整个测试流程，：
![执行流图](https://github.com/YapingZ/Pedestrian-behavior-trajectory-prediction/blob/master/picture/1.png)


