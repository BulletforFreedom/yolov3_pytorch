# -*- coding: utf-8 -*-
import torch as t
import torch.nn as nn
import numpy as np
#import math

from boxes.boxes import Boxes as bbox
from net.darknet import Darknet
import util as ut
from common.coco_dataset import COCODataset
from logger import Logger as log

class YOLO3Loss(nn.Module):
    def __init__(self, anchor_list, scaled_anchor_list, num_classes, inp_dim, stride, ignore_threshold=0.5):
        super(YOLO3Loss,self).__init__()
        self.anchors = anchor_list                 #type: int
        self.scaled_anchor_list=scaled_anchor_list #tpye: np.float34, len: 3
        self.num_feature_map = len(anchor_list)    #3
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = inp_dim    #416
        self.stride=-1

        self.ignore_threshold = ignore_threshold

        '''
        self.coord_scale = loss_lambda[0]
        self.object_scale = loss_lambda[1]
        self.noobject_scale=loss_lambda[2]
        self.class_scale = loss_lambda[3]
        '''

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, prediction, targets, stride):

        self.stride=stride
        bs = prediction.size(0)
        total_anchors = prediction.size(1) #10647[507,2028,8112]
        feature_map_size_list = [int((self.img_size / self.stride) * pow(2,i)) for i in range(self.num_feature_map)]
                
        # Get outputs
        x = prediction[..., 0]          # Center x
        y = prediction[..., 1]         # Center y
        w = prediction[..., 2]                         # Width
        h = prediction[..., 3]                         # Height
        conf = prediction[..., 4]       # Conf
        pred_cls = prediction[..., 5:]  # Cls pred.

        mask, noobj_mask, tx, ty, tw, th, weight_w_h, tconf, tcls = self.get_target(targets, total_anchors,
                                                                        bs, feature_map_size_list,
                                                                        self.ignore_threshold)
        mask, noobj_mask = mask.cuda(), noobj_mask.cuda()
        tx, ty, tw, th = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda()
        weight_w_h, tconf, tcls = weight_w_h.cuda(), tconf.cuda(), tcls.cuda()
        #  losses.
        loss_x = self.mse_loss(x * mask * weight_w_h, tx * weight_w_h)
        loss_y = self.mse_loss(y * mask * weight_w_h, ty * weight_w_h)
        loss_w = self.mse_loss(w * mask * weight_w_h, tw * weight_w_h)
        loss_h = self.mse_loss(h * mask * weight_w_h, th * weight_w_h)
        loss_conf = self.bce_loss(conf * mask, tconf) + \
            0.5 * self.bce_loss(conf * noobj_mask, noobj_mask * 0.0)
        loss_cls = self.bce_loss(pred_cls[mask == 1], tcls[mask == 1])
        #  total loss = losses * weight
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

        return loss#, loss_x.item(), loss_y.item(), loss_w.item(),\
            #loss_h.item(), loss_conf.item(), loss_cls.item()
        
    
    def get_target(self, target, total_anchors, bs, feature_map_size_list, ignore_threshold):
        '''
        noobj_mask: mask the non-object anchor boxes which ious are less than threshold
        mask: mark the object anchor boxes with best iou
        ignore the anchor boxes which ious are greater than threshold and less than best iou
        '''
        
        mask = t.zeros(bs, total_anchors, requires_grad=False)      #zeros[bs,10647]
        noobj_mask = t.ones(bs, total_anchors, requires_grad=False)    #ones[bs,10647]
        tx = t.zeros(bs, total_anchors, requires_grad=False)           #zeros[bs,10647]
        ty = t.zeros(bs, total_anchors, requires_grad=False)           #zeros[bs,10647]
        tw = t.zeros(bs, total_anchors, requires_grad=False)           #zeros[bs,10647]
        th = t.zeros(bs, total_anchors, requires_grad=False)           #zeros[bs,10647]
        weight_w_h = t.zeros(bs, total_anchors, requires_grad=False)   #zeros[bs,10647] 2-gw*gh
        tconf = t.zeros(bs, total_anchors, requires_grad=False)
        tcls = t.zeros(bs, total_anchors, self.num_classes, requires_grad=False)
                                                                                     #zeros[bs,10647,80]
        last_num_anchors_list=[0]
        for i in range(self.num_feature_map-1):
            last_num_anchors_list.append(last_num_anchors_list[i]+
                                         len(self.scaled_anchor_list[i]) * feature_map_size_list[i] ** 2)
        
        for b in range(bs):
            for n in range(target.shape[1]):
                if target[b, n].sum() == 0:
                    continue

                best_iou_list=[]
                best_iou_index_list=[]
                gi_j_list = []
                for m in range(self.num_feature_map):
                    # Convert to position relative to box
                    gx = target[b, n, 1] * feature_map_size_list[m]
                    gy = target[b, n, 2] * feature_map_size_list[m]
                    gw = target[b, n, 3] * feature_map_size_list[m]
                    gh = target[b, n, 4] * feature_map_size_list[m]
                    
                    # Get grid box indices
                    gi = int(gx)
                    gj = int(gy)
                    gi_j_list.append((gi,gj))
                    # Get shape of gt box
                    gt_box = t.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)                    
                    # Get shape of anchor box
                    anchor_shapes = t.FloatTensor(np.concatenate((np.zeros((len(self.scaled_anchor_list[m]), 2)),
                                                                  self.scaled_anchor_list[m]), 1))                    
                    # Calculate iou between gt and anchor shapes
                    anch_ious = bbox.bbox_iou(gt_box, anchor_shapes)
                    # Where the overlap is larger than threshold set mask to zero (ignore)
                    for i in range(anch_ious.size()[0]):
                        if anch_ious[i] > ignore_threshold:
                                noobj_mask[b, (last_num_anchors_list[m] + len(self.scaled_anchor_list[m]) * gi * gj)+i] = 0

                    # Find the best matching anchor box

                    best_iou = t.max(anch_ious)
                    best_index = t.argmax(anch_ious)
                    best_iou_list.append(best_iou)
                    best_iou_index_list.append(best_index)
                
                best_iou_feature_index=np.argmax(best_iou_list) #which feature map has total best iou
                best_iou_index=best_iou_index_list[best_iou_feature_index]

                # Masks
                mask[b, last_num_anchors_list[best_iou_feature_index]\
                         + len(self.scaled_anchor_list[m]) * gi_j_list[best_iou_feature_index][0] * gi_j_list[best_iou_feature_index][1]\
                         + best_iou_index] = 1                
                # Coordinates
                tx[b, last_num_anchors_list[best_iou_feature_index]\
                         + len(self.scaled_anchor_list[m]) * gi_j_list[best_iou_feature_index][0] * gi_j_list[best_iou_feature_index][1]\
                         + best_iou_index] = target[b, n, 1] * self.img_size
                ty[b, last_num_anchors_list[best_iou_feature_index]\
                         + len(self.scaled_anchor_list[m]) * gi_j_list[best_iou_feature_index][0] * gi_j_list[best_iou_feature_index][1]\
                         + best_iou_index] = target[b, n, 2] * self.img_size
                # Width and height
                tw[b, last_num_anchors_list[best_iou_feature_index]\
                         + len(self.scaled_anchor_list[m]) * gi_j_list[best_iou_feature_index][0] * gi_j_list[best_iou_feature_index][1]\
                         + best_iou_index] = target[b, n, 3] * self.img_size
                th[b, last_num_anchors_list[best_iou_feature_index]\
                         + len(self.scaled_anchor_list[m]) * gi_j_list[best_iou_feature_index][0] * gi_j_list[best_iou_feature_index][1]\
                         + best_iou_index] = target[b, n, 4] * self.img_size                
                weight_w_h[b, last_num_anchors_list[best_iou_feature_index]\
                         + len(self.scaled_anchor_list[m]) * gi_j_list[best_iou_feature_index][0] * gi_j_list[best_iou_feature_index][1]\
                         + best_iou_index] = 2-target[b, n, 3]*target[b, n, 4]
                tconf[b, last_num_anchors_list[best_iou_feature_index]\
                         + len(self.scaled_anchor_list[m]) * gi_j_list[best_iou_feature_index][0] * gi_j_list[best_iou_feature_index][1]\
                         + best_iou_index] = float(np.max(best_iou_list))
                # One-hot encoding of label
                tcls[b, last_num_anchors_list[best_iou_feature_index]\
                         + len(self.scaled_anchor_list[m]) * gi_j_list[best_iou_feature_index][0] * gi_j_list[best_iou_feature_index][1]\
                         + best_iou_index, int(target[b, n, 0])] = 1
                    
                    

        return mask, noobj_mask, tx, ty, tw, th, weight_w_h, tconf, tcls
        
if __name__ == '__main__':
    
    # DataLoader
    dataloader = t.utils.data.DataLoader(COCODataset("/home/lsk/Downloads/YOLOv3_PyTorch/data/coco/trainvalno5k.txt",
                                         (416, 416), is_training=True),
                                         batch_size=2, shuffle=True, num_workers=32, pin_memory=True)
    
    model = Darknet("../cfg/yolov3.cfg")
    #model.load_weights("../yolov3.weights")
    
    model.train(True)
    #model = nn.DataParallel(model)
    if t.cuda.is_available():        
        model.cuda()        
    loss_function = YOLO3Loss(model.anchor_list, model.scaled_anchor_list, model.num_classes, model.inp_dim, model.iou_threshold)

    # Start the training loop
    
    for epoch in range(1):
        for step, samples in enumerate(dataloader):
            images, labels = samples["image"], samples["label"]
            images=images.cuda()
            labels=labels.cuda()
            prediction = model(images)
            loss=loss_function(prediction, labels, model.stride)
            if step==0:
                break
    
    loss=loss.cpu().numpy()
    t.cuda.empty_cache()
    print(loss)
#/home/lsk/Downloads/YOLOv3_PyTorch/data/coco/images/train2014/COCO_train2014_000000000025.jpg