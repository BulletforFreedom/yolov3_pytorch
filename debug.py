#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 16:48:50 2018

@author: lsk
"""

# -*- coding: utf-8 -*-
import torch as t
import torch.nn as nn
import numpy as np
import cv2
#import math

from boxes.boxes import Boxes as bbox
from net.darknet import Darknet
from common.coco_dataset import COCODataset
from layer.loss_function import YOLO3Loss
import detector
import util as ut
from cfg.tools.configer import Configer
#from data_loader.det_data_loader import DetDataLoader as DataLoader

#class debug_truth(object):
    #def __init__(self):
        
if __name__=='__main__':
    
    # DataLoader
    dataloader = t.utils.data.DataLoader(COCODataset("/home/lsk/Downloads/YOLOv3_PyTorch/data/coco/trainvalno5k.txt",
                                         (416, 416), is_training=True),
                                         batch_size=1, shuffle=False, num_workers=32, pin_memory=True)
    
    cfg=Configer("./cfg/yolov3_test.cfg")
    net_info=cfg.get_net_info()
    blocks=cfg.get_blocks()
    
    model = Darknet(net_info, blocks)
    #model.load_weights("../yolov3.weights")
    
    model.train(True)
    #model = nn.DataParallel(model)
    if t.cuda.is_available():        
        model.cuda()      
        
    #dataloader = DataLoader(model.get_net_info())
        
    loss_function = YOLO3Loss(model.anchor_list, model.scaled_anchor_list, cfg.get_num_classes(), 
                              cfg.get_inp_dim(), cfg.get_iou_threshold())

    # Start the training loop
    
    for epoch in range(1):
        for step, samples in enumerate(dataloader):
            images, labels = samples["image"], samples["label"]
            images=images.cuda()            
            prediction = model(images)
            #loss=loss_function(prediction, labels, model.stride)
            labels=loss_function.debug_loss(prediction, labels, model.stride)
            labels=labels.cuda()
            DK=detector.DK_Output()
            result=DK.write_results(labels, cfg.get_num_classes())
            result=result.cpu().numpy()
            t.cuda.empty_cache()
            img_name='/home/lsk/Downloads/YOLOv3_PyTorch/data/coco/images/train2014/COCO_train2014_000000000009.jpg'
            im = cv2.imread(img_name)
            im_gt = im.copy()
            gt_boxes = [[x[1], x[2], x[3], x[4]] for x in result]
            gt_class_ids=[int(x[-1]) for x in result]
            im = ut.plot_bb(im_gt,gt_boxes,gt_class_ids,cfg.get_inp_dim())
            cv2.imwrite('001_new.jpg', im)
            '''
            for i, image in enumerate(images.cpu()):
                image = image.numpy()
                box_list=[]
                for l in result:
                    if l.sum() == 0 or int(l[0]) != i:
                        continue
                    box_list.append(l)
                gt_boxes = [[x[1], x[2], x[3], x[4]] for x in box_list]
                gt_class_ids=[int(x[-1]) for x in box_list]
                im = ut.plot_bb(image,gt_boxes,gt_class_ids,model.inp_dim)
                    #x1 = int((l[1] - l[3] / 2) * w)
                    #y1 = int((l[2] - l[4] / 2) * h)
                    #x2 = int((l[1] + l[3] / 2) * w)
                    #y2 = int((l[2] + l[4] / 2) * h)                    
                    #cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
                #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite("step{}_{}.jpg".format(step, i), image)
            '''
            if step==0:
                break