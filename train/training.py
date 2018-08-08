#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 10:35:06 2018

@author: lvsikai

Email: lyusikai@gmail.com
"""

import os
import sys
import time
import datetime
from math import isnan
import cv2

import torch as t
import torch.nn as nn
from torch.autograd import Variable

from net.darknet import Darknet
from layer.loss_function import YOLO3Loss
import detector
import util as ut
from cfg.tools.configer import Configer
from data_loader.det_data_loader import DetDataLoader as DataLoader
from logger import Logger as log


class Training(object):
    def __init__(self):
        self.cfg = Configer("../cfg/yolov3.cfg")
        
    def _get_optimizer(self, params):
        optimizer = None
        name = self.cfg.get_optimizer()
        if name == 'sgd':
            optimizer=t.optim.SGD(params, lr=self.cfg.get_learning_rate(),
                                 weight_decay=self.cfg.get_weight_decay())
        else:
            log.error('No optimizer!')
        return optimizer
        
    def __call__(self):
        start_time = time.strftime("%y-%m-%d %H:%M:%S", time.localtime())
        print(start_time)
        
        dataloader = DataLoader(self.cfg)
        training_data_loader = dataloader.get_loader()
        
        model = Darknet(self.cfg)        
        model.train(True)
        
        loss_function = YOLO3Loss(self.cfg)
        
        optimizer = self._get_optimizer(model.parameters())
        lr_scheduler = t.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=self.cfg.get_lr_steps(),
                                                   gamma=self.cfg.get_lr_gamma())
        
        model = nn.DataParallel(model)
        if t.cuda.is_available():        
            model.cuda()              
        
        start_training = time.time()
        total_itr = 0
        ten_batch_loss = []
        # Start the training loop    
        for epoch in range(self.cfg.get_epochs()):
            #log.info('Input image: %d'%self.cfg.get_resize_dim())
            for step, (images, gt_bboxes, gt_labels) in enumerate(training_data_loader):
                start_batch = time.time()
                total_itr += 1
    
                images=Variable(images,requires_grad=True).cuda()  
                
                prediction = model(images)
                loss, x, y, w, h, obj_conf, noobj_conf, cls \
                = loss_function(prediction, gt_labels, gt_bboxes)
                
                optimizer.zero_grad()           # clear gradients for this training step
                loss.backward()                 # backpropagation, compute gradients
                optimizer.step()
                
                ten_batch_loss.append(loss.item())
                if len(ten_batch_loss) > 10:
                    ten_batch_loss.pop(0)
                            
                if (step + 1) % 100 == 0:
                    print('loos_x: %.4f'%x)
                    print('loos_y: %.4f'%y)
                    print('loos_w: %.4f'%w)
                    print('loos_h: %.4f'%h)
                    print('loos_obj: %.4f'%obj_conf)
                    print('loos_noobj: %.4f'%noobj_conf)
                    print('loos_cls: %.4f'%cls) 
                    avg_loss = sum(ten_batch_loss)/10
                    print('Epoch: %d, Step: %d, Current_loss: %.4f, Avg_loss: %.4f, %.2f seconds\n' 
                          %(epoch+1, step+1, loss.item(), avg_loss, time.time()-start_batch))                
                
                if total_itr % 2000 == 0 and total_itr > 10000:
                    backup_dir = os.path.join(self.cfg.get_backup_path(),'%d_params.pkl'%total_itr)
                    log.info('Saving weights to %s'%backup_dir)
                    t.save(model.state_dict(), backup_dir)
                if epoch + 1 < self.cfg.get_epochs() and self.cfg.is_mul_train() and step == 199:
                    break
            lr_scheduler.step()
            if epoch + 2 < self.cfg.get_epochs() and self.cfg.is_mul_train():
                self.cfg.set_resize_dim(ut.random_resize_image(self.cfg.get_final_inp_dim(),self.cfg.get_total_strides()))
            else:
                self.cfg.set_resize_dim(self.cfg.get_final_inp_dim())        
            
        t.cuda.empty_cache()
        end_training = time.time()
        print(end_training - start_training)
        end_time = time.strftime("%y-%m-%d %H:%M:%S", time.localtime())
        print(end_time)
        
if __name__ == '__main__':
    trainer=Training()
    trainer()