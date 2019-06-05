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
##执行步骤
![avatar](/home/skk/1.png)

