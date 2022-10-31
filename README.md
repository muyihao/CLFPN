
A Context and Level Aware Feature Pyramid Network for Object Detection with Attention Mechanism
---------------------
By Hao Yang, Yi Zhang

This project is based on [mmdetection](https://github.com/open-mmlab/mmdetection)


Introduction
----------------
An object detection task includes classification and localization, which require large receptive field and high-resolution input respectively. How to strike a balance between the two conflicting needs remains a difficult problem in this field. Fortunately, feature pyramid network (FPN) realizes the fusion of low-level and high-level features, which alleviates this dilemma to some extent. However, existing FPN based networks overlooked the importance of features of different levels during fusion process. Their simple fusion strategies can easily cause overwritten of important information, leading to serious aliasing effect. In this paper, we propose an improved object detector based on context and level aware feature pyramid networks. Experiments have been conducted on mainstream datasets to validate the effectiveness of our network, where it exhibits superior performances than other state-of-the-art works.

Install
-------------
Please refer to [INSTALL.md](INSTALL.md) for installation.

**note**: In this project, we only uploaded the core configuration file and model, other files could be found in [mmdetection](https://github.com/open-mmlab/mmdetection).

Prepare data
----------
```
  mkdir -p data/coco
  ln -s /path_to_coco_dataset/annotations data/coco/annotations
  ln -s /path_to_coco_dataset/train2017 data/coco/train2017
  ln -s /path_to_coco_dataset/test2017 data/coco/test2017
  ln -s /path_to_coco_dataset/val2017 data/coco/val2017
```

Training
--------------
```shell
./tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --validate --work_dir <WORK_DIR>
```
For example,
```shell
./tools/dist_train.sh configs/faster_rcnn_r50_clfpn_1x_coco.py 8 --validate --work_dir faster_rcnn_r50_clfpn_1x
```

see more details at [mmdetection](https://github.com/open-mmlab/mmdetection)


Testing
-----------
```shell
python tools/test.py <CONFIG_FILE> <CHECKPOINT_FILE> --gpus <GPU_NUM> --out <OUT_FILE> --eval <EVAL_TYPE>
```
When test results of detection, use `--eval bbox`. When test results of instance segmentation, use `--eval bbox segm`. See more details at [mmdetection](https://github.com/open-mmlab/mmdetection).

For example,
```shell
python tools/test.py configs/faster_rcnn_r50_clfpn_1x_coco.py <CHECKPOINT_FILE> --gpus 8 --out results.pkl --eval bbox segm
```

Results on MS COCO testdev2017
---------
| Backbone | detector | schedule | mAP(det)  |
|----------|--------|-----------|-----------|
| ResNet-50 | Faster R-CNN | 1x | 39.2 |
| ResNet-101 | Faster R-CNN | 1x | 41.0 |
| ResNeXt-101-64x4d | Faster R-CNN | 1x | 43.3     |


License
--------
This project is released under the [Apache 2.0 license](LICENSE)
