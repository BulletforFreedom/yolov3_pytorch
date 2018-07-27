#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 16:48:50 2018

@author: lsk
"""
# -*- coding: utf-8 -*-
import os

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
from data_loader.det_data_loader import DetDataLoader as DataLoader

#class debug_truth(object):
    #def __init__(self):
        
if __name__=='__main__':
    
    # DataLoader
    '''
    dataloader1 = t.utils.data.DataLoader(COCODataset("/home/lsk/Downloads/YOLOv3_PyTorch/data/coco/trainvalno5k.txt",
                                         (416, 416), is_training=True),
                                         batch_size=1, shuffle=False, num_workers=32, pin_memory=True)
    for step, samples in enumerate(dataloader1):
        images, labels = samples["image"], samples["label"]
        print(images.size())
        print(labels.shape)
        break
    '''
    cfg=Configer("./cfg/yolov3_train.cfg")
    net_info=cfg.get_net_info()
    blocks=cfg.get_blocks()
    
    model = Darknet(cfg)
    #model.load_weights("../yolov3.weights")
    
    model.train(True)
    #model = nn.DataParallel(model)
    if t.cuda.is_available():        
        model.cuda()      
        
    dataloader = DataLoader(cfg)
        
    loss_function = YOLO3Loss(cfg)

    # Start the training loop
    
    train_loader = dataloader.get_loader()
    
    for epoch in range(1):
        for step, (images, bboxes, labels) in enumerate(train_loader):

            print(images.shape)
            print(bboxes[0].size())
            print(labels[0])
            break
            images=images.cuda()            
            prediction = model(images)
            #loss=loss_function(prediction, labels, model.stride)
            labels=loss_function.debug_loss(prediction, labels)
            labels=labels.cuda()
            DK=detector.DK_Output()
            results=DK.write_results(labels, cfg.get_num_classes())
            results=results.cpu().numpy()
            t.cuda.empty_cache()
            break
            img_names=['COCO_train2014_000000000009.jpg',
                      'COCO_train2014_000000000025.jpg',
                      'COCO_train2014_000000000030.jpg',
                      'COCO_train2014_000000000034.jpg',
                      'COCO_train2014_000000000036.jpg',
                      'COCO_train2014_000000000049.jpg',
                      'COCO_train2014_000000000061.jpg',
                      'COCO_train2014_000000000064.jpg',
                      'COCO_train2014_000000000071.jpg',
                      'COCO_train2014_000000000072.jpg']
            for i, img_name in enumerate(img_names):
                img_dir = os.path.join('/home/lsk/Downloads/YOLOv3_PyTorch/data/coco/images/train2014/',img_name)
                im = cv2.imread(img_dir)
                im_gt = im.copy()            
                gt_boxes = [[x[1], x[2], x[3], x[4]] for x in results if x[0]==i]
                gt_class_ids=[int(x[-1]) for x in results if x[0]==i]
                gt_class_name=[ cfg.get_dataset_name_seq()[x] for x in gt_class_ids]
                im = ut.plot_bb(im_gt,gt_boxes,gt_class_name,cfg.get_inp_dim())
                win_name = '1'
                cv2.imshow(win_name,im)
                cv2.moveWindow(win_name,10,10)
                
                k = cv2.waitKey(0)
                if k==ord('q'):
                    cv2.destroyAllWindows()
                    break
                elif k==ord('c'):
                    try:
                        cv2.destroyWindow(win_name)
                    except:
                        cv2.destroyAllWindows()
                        break
            #cv2.imwrite('001_new.jpg', im)
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