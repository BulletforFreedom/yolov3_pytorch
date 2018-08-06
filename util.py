from logger import Logger as log

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable

import numpy as np
import cv2 
import matplotlib.pyplot as plt
import random
#from bbox import bbox_iou


def unique(tensor1d):
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


def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

def get_test_input(img_name,inp_dim):
    img = cv2.imread(img_name)
    img = cv2.resize(img, (inp_dim,inp_dim)) #Resize to the input dimension
    img_ = img[:,:,::-1].transpose((2,0,1)) # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0 #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float() #Convert to float
    img_ = Variable(img_) # Convert to Variable
    return img_

def plot_bb(im,bb,cls,inp_dim,textSize=1,textThickness=2):
    if im.shape[0]==3:
        im=im.transpose(1,2,0)
    h,w,c = im.shape

    for idx,box in enumerate(bb):
        
        x1 = int(max([0, (box[0]/inp_dim)*w ]))
        x2 = int(min([w, (box[2]/inp_dim)*w ]))
        y1 = int(max([0, (box[1]/inp_dim)*h ]))
        y2 = int(min([h, (box[3]/inp_dim)*h ]))        

        cv2.rectangle(im,(x1,y1),(x2,y2),(0,0,255),1)
        cv2.putText(im,
                    '%s'%(cls[idx]),
                    (x1,int(y1*1.1)),
                    cv2.FONT_HERSHEY_SIMPLEX,textSize,
                    (0,200,255),
                    thickness=textThickness)    
    return im

def random_resize_image(original_img_size, strides):
    amplitude = original_img_size // 32 -5         
    new_size = (random.randint(0,9) + amplitude) * strides
    return new_size
        

if __name__ == '__main__':
    cfgfile='./cfg/yolov3.cfg'
    print(111)    