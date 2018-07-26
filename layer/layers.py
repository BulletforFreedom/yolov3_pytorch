# -*- coding: utf-8 -*-
import util as ut

import numpy as np
import cv2 
import matplotlib.pyplot as plt

import torch as t
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from logger import Logger as log

class EmptyLayer(nn.Module):
    def __init__(self,shortcut_from=-1,route_start=-1,route_end=-1,anchors=[]):
        super(EmptyLayer,self).__init__()
        self.shortcut_from=shortcut_from
        self.route_start=route_start
        self.route_end=route_end
        self.anchors=anchors
    
    def get_shortcut_from(self):
        return self.shortcut_from
    
    def get_route_parm(self):
        return self.route_start,self.route_end
    
    def get_anchors(self):
        return self.anchors
     
class DetectionLayer(nn.Module):
    def __init__(self, anchors, configer, CUDA=True):
        super(DetectionLayer,self).__init__()
        self.anchors=anchors
        self.configer=configer
        self.CUDA=CUDA
        self.stride=-1
        self.scaled_anchors=[]
    
    def forward(self, x):
        x = x.data
        #global CUDA
        prediction = self.predict_transform(x)
        return prediction

    def predict_transform(self,prediction):
        #inp_dim: weith or height of input image
        #stride: Scaling ratio
        batch_size = prediction.size(0)
        self.stride=self.configer.get_inp_dim()//prediction.size(2)
        grid_size=prediction.size(2)
        bbox_attrs = 5 + self.configer.get_num_classes()
        num_anchors = len(self.anchors)
        
        if len(self.scaled_anchors)==0:
            self.scaled_anchors = [(a[0]/self.stride, a[1]/self.stride) for a in self.anchors]
            self.configer.set_scaled_anchor_list(self.scaled_anchors)
            #log space transform height and the width
            self.scaled_anchors=t.FloatTensor(self.scaled_anchors)
            if self.CUDA:
                self.scaled_anchors=self.scaled_anchors.cuda()
            self.scaled_anchors=self.scaled_anchors.repeat(grid_size*grid_size,1).unsqueeze(0)
        
        prediction = prediction.view(batch_size,bbox_attrs*num_anchors,grid_size*grid_size)
        prediction = prediction.transpose(1,2).contiguous()
        prediction = prediction.view(batch_size,grid_size*grid_size*num_anchors,bbox_attrs)
        
        #Sigmoid the centre_X, centre_Y
        prediction[:,:,:2] = t.sigmoid(prediction[:,:,:2])    
        #Add the center offsets
        grid_len = np.arange(grid_size)
        a,b=np.meshgrid(grid_len,grid_len)
        x_offset=t.FloatTensor(a).view(-1,1)
        y_offset=t.FloatTensor(b).view(-1,1)
        if self.CUDA:
            x_offset=x_offset.cuda()
            y_offset=y_offset.cuda()
        x_y_offset=t.cat((x_offset,y_offset),1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
        prediction[:,:,:2] += x_y_offset
        
        
        prediction[:,:,2:4]=t.exp(prediction[:,:,2:4])*self.scaled_anchors 
        
        prediction[:,:,:4] *= self.stride
        
        #Sigmoid the object confidencce
        prediction[:,:,4] = t.sigmoid(prediction[:,:,4])
        #Softmax the class scores
        prediction[:,:,5: 5 + self.configer.get_num_classes()] = t.sigmoid(prediction[:,:, 5 : 5 + self.configer.get_num_classes()])
        
        return prediction
