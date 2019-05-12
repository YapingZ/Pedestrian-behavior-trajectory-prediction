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
> ```python train_social_model.py --config path/to/comfig.json [--out_root OUT_ROOT]```

## Testing
Run evaluate_social_model.py
> ```python train_social_model.py --trained_model_config path/to/config.json --trained_model_file path/to/trained_model.h5```

## Restrictions
- work only on batch size = 1
- require much RAM (use almost all 16GB in my environment)
