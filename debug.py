#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 16:48:50 2018

@author: lsk
"""
# -*- coding: utf-8 -*-
import torch as t
import cv2
#import math

from net.darknet import Darknet
from layer.loss_function import YOLO3Loss
import detector
import util as ut
from cfg.tools.configer import Configer
from data_loader.det_data_loader import DetDataLoader as DataLoader
        
if __name__=='__main__':    
    
    cfg=Configer("./cfg/yolov3.cfg")
    net_info=cfg.get_net_info()
    blocks=cfg.get_blocks()
    
    model = Darknet(cfg)
    
    model.train(True)
    #model = nn.DataParallel(model)
    if t.cuda.is_available():        
        model.cuda()      
        
    dataloader = DataLoader(cfg, True)
    debug_loader = dataloader.get_loader()
        
    loss_function = YOLO3Loss(cfg)

    # Start the training loop    
    for epoch in range(cfg.get_epochs()):
        for step, (images, img_dir_list, gt_bboxes, gt_labels) in enumerate(debug_loader):

            images=images.cuda()            
            prediction = model(images)
            #loss=loss_function(prediction, labels, model.stride)
            origin_results=loss_function.debug_loss(prediction, gt_labels, gt_bboxes)
            origin_results=origin_results.cuda()
            DK=detector.DK_Output()
            results=DK.write_results(origin_results, cfg.get_num_classes())
            results=results.cpu().numpy()
            t.cuda.empty_cache()
            
            for i, img_dir in enumerate(img_dir_list):
                im = cv2.imread(img_dir)
                im_gt = im.copy()            
                gt_boxes = [[x[1], x[2], x[3], x[4]] for x in results if x[0]==i]
                gt_class_ids=[int(x[-1]) for x in results if x[0]==i]
                gt_class_name=[ cfg.get_dataset_name_seq()[x] for x in gt_class_ids]
                im = ut.plot_bb(im_gt,gt_boxes,gt_class_name,cfg.get_resize_dim())
                win_name = '1'
                cv2.imshow(win_name,im)
                cv2.moveWindow(win_name,10,10)
                
                k = cv2.waitKey(0)
                if k==ord('q'):
                    cv2.destroyAllWindows()
                elif k==ord('c'):
                    try:
                        cv2.destroyWindow(win_name)
                    except:
                        cv2.destroyAllWindows()
                        break 
            if epoch + 1 < cfg.get_epochs() and step == 2:
                break
            if step==3:
                break
        if epoch + 2 < cfg.get_epochs():
            cfg.reset_mul_train(1)
        else:
            cfg.reset_mul_train(2)
        cfg.reset_scaled_anchor_list()