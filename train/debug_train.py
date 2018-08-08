# -*- coding: utf-8 -*-

#torch.optim.lr_scheduler.ReduceLROnPlateau，LambdaLR，StepLR，MultiStepLR，ExponentialLR
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


if __name__ == "__main__":
    start_time = time.strftime("%y-%m-%d %H:%M:%S", time.localtime())
    print(start_time)
    cfg=Configer("../cfg/yolov3.cfg")
    net_info=cfg.get_net_info()
    blocks=cfg.get_blocks()
    
    model = Darknet(cfg)
    
    model.train(True)
    model = nn.DataParallel(model)
    if t.cuda.is_available():        
        model.cuda()      
        
    dataloader = DataLoader(cfg, True)
    debug_loader = dataloader.get_loader()
        
    loss_function = YOLO3Loss(cfg)
    optimizer = t.optim.SGD(model.parameters(), lr=cfg.get_learning_rate(),
                             weight_decay=cfg.get_weight_decay())
    #lr_scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1)
    DK=detector.DK_Output()
    start_training = time.time()
    total_itr = 0
    ten_batch_loss = []
    # Start the training loop    
    for epoch in range(cfg.get_epochs()):
        #log.info('Input image: %d'%cfg.get_resize_dim())
        for step, (images, img_dir_list, gt_bboxes, gt_labels) in enumerate(debug_loader):
            start_batch = time.time()
            total_itr += 1

            images=Variable(images,requires_grad=True).cuda()  
            
            prediction = model(images)
            loss, x, y, w, h, obj_conf, noobj_conf, cls \
            = loss_function(prediction, gt_labels, gt_bboxes)
            
            if isnan(loss):
                sys.exit()
            
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()  
            
            ten_batch_loss.append(loss.item())
            if len(ten_batch_loss) > 10:
                ten_batch_loss.pop(0)
                        
            if (step + 1) % 1 == 0:                
                print('loos_x: %.4f'%x)
                print('loos_y: %.4f'%y)
                print('loos_w: %.4f'%w)
                print('loos_h: %.4f'%h)
                print('loos_obj: %.4f'%obj_conf)
                print('loos_noobj: %.4f'%noobj_conf)
                print('loos_cls: %.4f'%cls) 
                avg_loss = sum(ten_batch_loss)/len(ten_batch_loss)
                print('Epoch: %d, Step: %d, Current_loss: %.4f, Avg_loss: %.4f, %.2f seconds\n' 
                      %(epoch+1, step+1, loss.item(), avg_loss, time.time()-start_batch))
            if total_itr % 10 == 0 and total_itr > 5:
                backup_dir = os.path.join(cfg.get_backup_path(),'%d_params.pkl'%total_itr)
                log.info('Saving weights to %s'%backup_dir)
                t.save(model.state_dict(), backup_dir)
            if epoch + 1 < cfg.get_epochs() and step == 0:
                break
        #lr_scheduler.step()
        if epoch + 2 < cfg.get_epochs() and cfg.is_mul_train():
            cfg.set_resize_dim(ut.random_resize_image(cfg.get_final_inp_dim(),cfg.get_total_strides()))
        else:
            cfg.set_resize_dim(cfg.get_final_inp_dim())        
        
    t.cuda.empty_cache()
    end_training = time.time()
    print(end_training - start_training)
    end_time = time.strftime("%y-%m-%d %H:%M:%S", time.localtime())
    print(end_time)
#time.strftime("%y-%m-%d %H:%M:%S", time.localtime())+' '+str(datetime.datetime.now().microsecond)