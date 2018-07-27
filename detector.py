from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
import argparse
import os 
import os.path as osp
#from darknet import Darknet
#from preprocess import prep_image, inp_to_image
import pandas as pd
import random 
import pickle as pkl
import itertools

import util as ut
from boxes.boxes import Boxes as bbox

class DK_Output:        
    
    def _unique(self,tensor1d):
        temp=[]
        for x in tensor1d:
            #log.info(x)
            if len(temp)==0:
                temp.append(x)
            else:
                for y in temp:
                    #log.info(y)
                    if x==y:
                        break
                    if y==temp[-1]:
                        temp.append(x)
        return temp
    

    
    def write_results(self, prediction, num_classes, confidence=0.5, nms_conf = 0.7):
        conf_mask=(prediction[:,:,4]>confidence).float().unsqueeze(2)
        prediction=prediction*conf_mask
        
        batch_size = prediction.size(0)
        box_a = torch.zeros(batch_size, prediction.size(1),4)
        box_a[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)    #top_left_x
        box_a[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)    #top_left_y
        box_a[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2)    #down_right_x
        box_a[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)    #down_right_x
        prediction[:,:,:4] = box_a[:,:,:4]
        
        write = False
        
        for i in range(batch_size):
            #image_pred = prediction[i]
            
            #Get the class having maximum score, and the index of that class
            max_conf, max_conf_index = torch.max(prediction[i][:,5:5+ num_classes], 1)
            max_conf = max_conf.float().unsqueeze(1)
            max_conf_index = max_conf_index.float().unsqueeze(1)
            seq = (prediction[i][:,:5], max_conf, max_conf_index)
            image_pred = torch.cat(seq, 1)#(x_left_up,y_left_up,x_right_down,y_right_down,max_conf,max_conf_index)
            
            #Get rid of the zero entries
            non_zero_ind =  torch.nonzero(image_pred[:,4]).squeeze()
            try:
                image_pred_ = image_pred[non_zero_ind,:]
            except:
                continue #if there is no result,
            try:
                img_classes=self._unique(image_pred_[:,-1])
            except:
                seq = torch.FloatTensor([i]).cuda(), image_pred_
                if not write:
                    output = torch.cat(seq).unsqueeze(0)
                    write = True
                else:
                    out = torch.cat(seq).unsqueeze(0)
                    output = torch.cat((output,out))                
                continue
            
            for cls in img_classes:#get the detections with one particular class
                
                cls_mask=image_pred_*(image_pred_[:,-1]==cls).float().unsqueeze(1)
                class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
                image_pred_class = image_pred_[class_mask_ind].view(-1,7)
                #sort the detections such that the entry with the maximum objectness
                #confidence is at the top                
                conf_sort_index=torch.sort(image_pred_class[:,4],descending=True)[1]
                image_pred_class=image_pred_class[conf_sort_index]
                #num = image_pred_class.size(0)#Number of detections
                if nms_conf !=0:#and image_pred_class.size(0)>1
                    image_pred_class=bbox.get_nms(image_pred_class,nms_conf)
                    
                #Concatenate the batch_id of the image to the detection
                #this helps us identify which image does the detection correspond to 
                #We use a linear straucture to hold ALL the detections from the batch
                #the batch_dim is flattened
                #batch is identified by extra batch column
                
                
                batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(i)#Which image does it belong to?
                seq = batch_ind, image_pred_class
                if not write:
                    output = torch.cat(seq,1)
                    write = True
                else:
                    out = torch.cat(seq,1)
                    output = torch.cat((output,out))
        return output

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    parser.add_argument("--scales", dest = "scales", help = "Scales to use for detection",
                        default = "1,2,3", type = str)
    
    return parser.parse_args()

if __name__ ==  '__main__':
    args = arg_parse()
    
    scales = args.scales
    
    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0

    CUDA = torch.cuda.is_available()
    
    classes = ut.load_classes('data/coco.names')