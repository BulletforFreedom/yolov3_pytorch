#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 16:24:49 2018
@author: lsk
"""

#from logger import Logger as log
import util as ut

import numpy as np
import cv2 
import matplotlib.pyplot as plt

import torch as t
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from logger import Logger as log

from layer.layers import EmptyLayer
from layer.layers import DetectionLayer
import detector


class Darknet(nn.Module):
    '''
    forward return:
        size(batch_size,num_anchor_boxes,x+y+w+h+c+cls)
    '''
    def __init__(self, configer):
        super(Darknet, self).__init__()
        self.anchor_list=[]
        self.configer=configer
        self.blocks = self.configer.get_blocks()
        self.net_info = self.configer.get_net_info()
        self.module_list = self.create_modules()        
        self.stride=-1
        self.flag=0
    
    
    def forward(self, x):
        detections=[]
        modules=[x['type'] for x in self.blocks]
        outputs={}
        write=0
        if len(self.configer.get_scaled_anchor_list())!=0:            
            self.flag=1
        
        for i in range(len(modules)):
            if modules[i]=='convolutional' or modules[i]=='upsample':
                x=self.module_list[i](x)
                   
                outputs[i]=x
                
            elif modules[i]=='shortcut':
                from_=self.module_list[i][0].shortcut_from
                x=outputs[i-1]+outputs[from_]
                outputs[i]=x
                
            elif modules[i]=='route':
                start = self.module_list[i][0].route_start
                end = self.module_list[i][0].route_end
                x=outputs[start]
                if end>0:
                    x=t.cat((x,outputs[end]),1)
                outputs[i]=x
            elif modules[i]=='yolo':

                x = self.module_list[i](x)#ut.predict_transform(x, self.inp_dim, anchors, self.num_classes, CUDA)
                #if not self.flag:   
                    #self.configer.set_scaled_anchor_list(self.module_list[i][0].scaled_anchors.cpu().numpy())
                if type(x) == int:#?
                    continue
                
                if not write:
                    detections=x
                    if self.stride==-1:
                        self.stride=self.module_list[i][0].stride
                    write=1
                else:
                    detections = t.cat((detections, x), 1)
                outputs[i] = outputs[i-1]
        try:
            return detections
        except:
            return 0

    def load_weights(self, weightfile):
        
        #Open the weights file
        fp = open(weightfile, "rb")

        #The first 4 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4. IMages seen 
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = t.from_numpy(header)
        self.seen = self.header[3]
        
        #The rest of the values are the weights
        # Let's load them up
        weights = np.fromfile(fp, dtype = np.float32)
        
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
            
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
                
                conv = model[0]
                
                if (batch_normalize):
                    bn = model[1]
                    
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
                    
                    #Load the weights
                    bn_biases = t.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    
                    bn_weights = t.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    
                    bn_running_mean = t.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    
                    bn_running_var = t.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = t.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                    
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                    
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                    
                    
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                
                #Do the same as above for weights
                conv_weights = t.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
                
    def save_weights(self, savedfile, cutoff = 0):
            
        if cutoff <= 0:
            cutoff = len(self.blocks) - 1
        
        fp = open(savedfile, 'wb')
        
        # Attach the header at the top of the file
        self.header[3] = self.seen
        header = self.header

        header = header.numpy()
        header.tofile(fp)
        
        # Now, let us save the weights 
        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]["type"]
            
            if (module_type) == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
                    
                conv = model[0]

                if (batch_normalize):
                    bn = model[1]
                
                    #If the parameters are on GPU, convert them back to CPU
                    #We don't convert the parameter to GPU
                    #Instead. we copy the parameter and then convert it to CPU
                    #This is done as weight are need to be saved during training
                    ut.cpu(bn.bias.data).numpy().tofile(fp)
                    ut.cpu(bn.weight.data).numpy().tofile(fp)
                    ut.cpu(bn.running_mean).numpy().tofile(fp)
                    ut.cpu(bn.running_var).numpy().tofile(fp)
                
            
                else:
                    ut.cpu(conv.bias.data).numpy().tofile(fp)
                
                
                #Let us save the weights for the Convolutional layers
                ut.cpu(conv.weight.data).numpy().tofile(fp)
    
    
    def create_modules(self):
        module_list=nn.ModuleList()
        
        index=0
        
        prev_filters=3
        output_filters=[]
        
        for x in self.blocks:
            module=nn.Sequential()            
            
            if x['type']=='convolutional':
                try:
                    batch_normalize=int(x['batch_normalize'])
                    bias=False
                except:
                    batch_normalize=0
                    bias=True
                
                filters= int(x["filters"])
                padding = int(x["pad"])
                kernel_size = int(x["size"])
                conv_stride = int(x["stride"])
                if padding:
                    padding=(kernel_size-1)//2
                else:
                    padding=0            
                conv=nn.Conv2d(prev_filters, filters, kernel_size, stride=conv_stride, padding=padding,bias=bias)
                module.add_module('cov_{0}'.format(index),conv)
                
                if batch_normalize:
                    bn=nn.BatchNorm2d(filters)
                    module.add_module('bn_{0}'.format(index),bn)
                if x['activation']=='leaky':
                    activn=nn.LeakyReLU(0.1,True)
                    module.add_module('activn_{0}'.format(index),activn)
                    
            elif x['type']=='shortcut':
                from_=int(x['from'])
                if from_<0:
                    from_=from_+index#must be positive
                shortcut=EmptyLayer(shortcut_from=from_)
                module.add_module('shortcut_{0}'.format(index),shortcut)
                
            elif x['type']=='softmax':
                softmax=nn.Softmax2d()
                module.add_module('softmax_{0}'.format(index),softmax)
                
            elif x['type']=='route':
                x['layers'] = x["layers"].split(',')
                start=int(x['layers'][0])
                if len(x['layers'])==1:
                    end=0
                elif len(x['layers'])==2:
                    end=int(x['layers'][1])
                else:
                    log.info("Something I dunno")
                    assert False 
                    
                #Negative anotation
                if start < 0: 
                    start = start + index
                
                if end < 0:
                    end = end + index
                
                route=EmptyLayer(route_start=start,route_end=end)#must be positive
                module.add_module('route_{0}'.format(index),route)
                if end>0:
                    filters=output_filters[start]+output_filters[end]
                else:
                    filters=output_filters[start]
                    
            elif x['type']=='upsample':
                upsample_stride=int(x['stride'])
                upsample=nn.Upsample(scale_factor=upsample_stride, mode='bilinear', align_corners=True)
                module.add_module('upsample_{0}'.format(index),upsample)
                
            elif x['type']=='yolo':
                mask=x['mask'].split(',')
                mask=[int(x) for x in mask]
                
                anchors=self.net_info['anchors'].split(',')
                anchors=[float(x) for x in anchors]
                anchors=[(anchors[x],anchors[x+1]) for x in range(0,len(anchors),2)]
                anchors=[anchors[x] for x in mask]            
                
                self.anchor_list.append(anchors)
                
                detection = DetectionLayer(anchors=anchors, 
                                           configer=self.configer,
                                           CUDA=t.cuda.is_available())
                module.add_module("Detection_{}".format(index), detection)
                
            else:
                log.info("Something I dunno")
                assert False            
                
            module_list.append(module)
            prev_filters = filters
            output_filters.append(filters)
            index+=1
        return module_list


if __name__=='__main__':
    
    model = Darknet("../cfg/yolov3.cfg")
    model.load_weights("../yolov3.weights")
    
    img_name="/home/lsk/Downloads/yolov3_pytorch/det_messi.jpg"
    inp = ut.get_test_input(img_name,model.inp_dim)
    CUDA = t.cuda.is_available()
    if CUDA:
        inp=inp.cuda()
        model.cuda()
    model.eval()
    prediction = model(inp)
    DK=detector.DK_Output()
    result=DK.write_results(prediction,model.get_num_classes())
    result=result.cpu().numpy()
    t.cuda.empty_cache()
    
    im = cv2.imread(img_name)
    #res=cv2.resize(im,(416,416),interpolation=cv2.INTER_CUBIC)
    im_gt = im.copy()
    gt_boxes = [[x[1], x[2], x[3], x[4]] for x in result]
    gt_class_ids=[int(x[-1]) for x in result]
    im = ut.plot_bb(im_gt,gt_boxes,gt_class_ids,model.inp_dim)
    cv2.imwrite('001_new.jpg', im)