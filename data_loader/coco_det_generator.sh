#!/usr/bin/env bash
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Generate train & val data.


export PYTHONPATH='/home/donny/Projects/PytorchCV'


COCO_TRAIN_IMG_DIR='/home/lsk/Downloads/YOLOv3_PyTorch/data/coco/images/train2014'
COCO_VAL_IMG_DIR='/home/lsk/Downloads/YOLOv3_PyTorch/data/coco/images/val2014'

COCO_ANNO_DIR='/home/lsk/Downloads/YOLOv3_PyTorch/data/coco/images/annotations/'
TRAIN_ANNO_FILE=${COCO_ANNO_DIR}'instances_train2014.json'
VAL_ANNO_FILE=${COCO_ANNO_DIR}'instances_val2014.json'

TRAIN_SAVE_DIR='/home/lsk/Downloads/yolov3_pytorch/data/COCO_DET/train'
VAL_SAVE_DIR='/home/lsk/Downloads/yolov3_pytorch/data/COCO_DET/val'


#python coco_det_generator.py --save_dir $TRAIN_SAVE_DIR \
#                             --anno_file $TRAIN_ANNO_FILE \
#                             --ori_img_dir $COCO_TRAIN_IMG_DIR

python coco_det_generator.py --save_dir $VAL_SAVE_DIR \
                             --anno_file $VAL_ANNO_FILE \
                             --ori_img_dir $COCO_VAL_IMG_DIR