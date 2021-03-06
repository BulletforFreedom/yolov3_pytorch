# -*- coding: utf-8 -*-
import torch as t
import torch.nn as nn
import numpy as np
import math

from boxes.boxes import Boxes as bbox
from net.darknet import Darknet
from common.coco_dataset import COCODataset
from cfg.tools.configer import Configer
from data_loader.det_data_loader import DetDataLoader as DataLoader

class YOLO3Loss(nn.Module):
    def __init__(self, configer):
        super(YOLO3Loss,self).__init__()
        self.configer=configer 
        
        #self.num_feature_map = len(self.configer.get_scaled_anchor_list())    #3
        self.num_classes = self.configer.get_num_classes()
        self.bbox_attrs = 5 + self.num_classes
        self.num_feature_map = -1
        self.scaled_anchor_list = -1

        loss_lambda = configer.get_loss_lambda()
        self.coord_scale = loss_lambda[0]
        self.object_scale = loss_lambda[1]
        self.noobject_scale=loss_lambda[2]
        self.class_scale = loss_lambda[3]
        

        self.mse_loss = nn.MSELoss(size_average=False)
        self.bce_loss = nn.BCELoss(size_average=False)

    def forward(self, prediction, labels, bboxes):
        
        stride=self.configer.get_total_strides()
        img_size=self.configer.get_resize_dim()
        num_feature_map = self.configer.get_num_feature_map()

        bs = prediction.size(0)
        total_anchors = prediction.size(1) #10647[507,2028,8112]
        feature_map_size_list = [((img_size // stride) * pow(2,i)) for i in range(num_feature_map)]
                
        # Get outputs
        x = prediction[..., 0]          # Center x
        y = prediction[..., 1]         # Center y
        w = prediction[..., 2]                         # Width
        h = prediction[..., 3]                         # Height
        conf = prediction[..., 4]       # Conf
        pred_cls = prediction[..., 5:]  # Cls pred.

        mask, noobj_mask, tx, ty, tw, th, weight_w_h, tconf, tcls = self._get_target(labels, bboxes, total_anchors,
                                                                        bs, img_size, num_feature_map, feature_map_size_list)
        noobj_mask = noobj_mask.cuda()
        #noobj_sum = noobj_mask.sum()
        loss_noobj_conf = 0.4 * self.bce_loss(conf * noobj_mask, noobj_mask * 0.0) / bs #/ noobj_sum                                      
        if t.nonzero(mask).numel() == 0:  #no object, all background                                                     
            return loss_noobj_conf, 0, 0, 0, 0, loss_noobj_conf, 0   
        
        mask = mask.cuda()
        #obj_sum = mask.sum()
        tx, ty, tw, th = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda()
        weight_w_h, tconf, tcls = weight_w_h.cuda(), tconf.cuda(), tcls.cuda()        
        #  losses        
        loss_x = self.mse_loss(x * mask * weight_w_h, tx * weight_w_h) / bs
        loss_y = self.mse_loss(y * mask * weight_w_h, ty * weight_w_h) / bs#* weight_w_h
        loss_w = self.mse_loss(w * mask * weight_w_h, tw * weight_w_h) / bs
        loss_h = self.mse_loss(h * mask * weight_w_h, th * weight_w_h) / bs
        loss_obj_conf = self.bce_loss(conf * mask, tconf) / bs#/ obj_sum
        loss_cls = self.bce_loss(pred_cls[mask == 1], tcls[mask == 1]) / (bs * self.num_classes)
        #  total loss = losses * weight
        loss = 2*(loss_x + loss_y + loss_w + loss_h) + loss_obj_conf + loss_noobj_conf + loss_cls

        return loss, loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(),\
               loss_obj_conf.item(), loss_noobj_conf.item(), loss_cls.item()
        
    
    def _get_target(self, labels, bboxes, total_anchors, bs, img_size, num_feature_map, feature_map_size_list):
        '''
        noobj_mask: mask the non-object anchor boxes which ious are less than threshold
        mask: mark the object anchor boxes with best iou
        ignore the anchor boxes which ious are greater than threshold and less than best iou
        '''
        if self.scaled_anchor_list == -1:
            self.scaled_anchor_list = self.configer.get_scaled_anchor_list() #len: 3
        
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
        for i in range(num_feature_map-1):
            last_num_anchors_list.append(last_num_anchors_list[i]+
                                         len(self.scaled_anchor_list[i]) * feature_map_size_list[i] ** 2)
        
        for b in range(bs):
            for n, label in enumerate(labels[b]):                
                
                best_iou_list=[]
                best_iou_index_list=[]
                gi_j_list = []
                for m in range(num_feature_map):
                    # Convert to position relative to box
                    gx = bboxes[b][n][0] * feature_map_size_list[m]
                    gy = bboxes[b][n][1] * feature_map_size_list[m]
                    gw = bboxes[b][n][2] * feature_map_size_list[m]
                    gh = bboxes[b][n][3] * feature_map_size_list[m]
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
                        if anch_ious[i] > self.configer.get_iou_threshold():
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
                         + len(self.scaled_anchor_list[m]) * (gi_j_list[best_iou_feature_index][0] + gi_j_list[best_iou_feature_index][1] * feature_map_size_list[best_iou_feature_index])\
                         + best_iou_index] = 1                
                # Coordinates
                tx[b, last_num_anchors_list[best_iou_feature_index]\
                         + len(self.scaled_anchor_list[m]) * (gi_j_list[best_iou_feature_index][0] + gi_j_list[best_iou_feature_index][1] * feature_map_size_list[best_iou_feature_index])\
                         + best_iou_index] = \
                   (bboxes[b][n][0] * feature_map_size_list[best_iou_feature_index]) % 1
                ty[b, last_num_anchors_list[best_iou_feature_index]\
                         + len(self.scaled_anchor_list[m]) * (gi_j_list[best_iou_feature_index][0] + gi_j_list[best_iou_feature_index][1] * feature_map_size_list[best_iou_feature_index])\
                         + best_iou_index] =\
                   (bboxes[b][n][1] * feature_map_size_list[best_iou_feature_index]) % 1 
                # Width and height
                tw[b, last_num_anchors_list[best_iou_feature_index]\
                         + len(self.scaled_anchor_list[m]) * (gi_j_list[best_iou_feature_index][0] + gi_j_list[best_iou_feature_index][1] * feature_map_size_list[best_iou_feature_index])\
                         + best_iou_index] = \
                   math.log((bboxes[b][n][2] * feature_map_size_list[best_iou_feature_index])/(self.scaled_anchor_list[best_iou_feature_index][best_iou_index][0])+ 1e-16)
                th[b, last_num_anchors_list[best_iou_feature_index]\
                         + len(self.scaled_anchor_list[m]) * (gi_j_list[best_iou_feature_index][0] + gi_j_list[best_iou_feature_index][1] * feature_map_size_list[best_iou_feature_index])\
                         + best_iou_index] = \
                   math.log((bboxes[b][n][3] * feature_map_size_list[best_iou_feature_index])/(self.scaled_anchor_list[best_iou_feature_index][best_iou_index][1])+ 1e-16)
                
                weight_w_h[b, last_num_anchors_list[best_iou_feature_index]\
                         + len(self.scaled_anchor_list[m]) * (gi_j_list[best_iou_feature_index][0] + gi_j_list[best_iou_feature_index][1] * feature_map_size_list[best_iou_feature_index])\
                         + best_iou_index] = 2-bboxes[b][n][2]*bboxes[b][n][3]                
                tconf[b, last_num_anchors_list[best_iou_feature_index]\
                         + len(self.scaled_anchor_list[m]) * (gi_j_list[best_iou_feature_index][0] + gi_j_list[best_iou_feature_index][1] * feature_map_size_list[best_iou_feature_index])\
                         + best_iou_index] = 1#float(np.max(best_iou_list))
                # One-hot encoding of label
                tcls[b, last_num_anchors_list[best_iou_feature_index]\
                         + len(self.scaled_anchor_list[m]) * (gi_j_list[best_iou_feature_index][0] + gi_j_list[best_iou_feature_index][1] * feature_map_size_list[best_iou_feature_index])\
                         + best_iou_index, label] = 1
                    
                    

        return mask, noobj_mask, tx, ty, tw, th, weight_w_h, tconf, tcls
        
    def debug_loss(self, prediction, labels, bboxes):
        img_size = self.configer.get_resize_dim()
        num_feature_map = self.configer.get_num_feature_map()
        stride=self.configer.get_total_strides()
        bs = prediction.size(0)
        total_anchors = prediction.size(1) #10647[507,2028,8112]
        feature_map_size_list = [((img_size // stride) * pow(2,i)) for i in range(num_feature_map)] 
        
        mask, noobj_mask, tx, ty, tw, th, weight_w_h, tconf, tcls = self._get_target(labels, bboxes, total_anchors,
                                                                        bs, img_size, num_feature_map, feature_map_size_list)
        tx = tx.unsqueeze(2)
        ty = ty.unsqueeze(2)
        tw = tw.unsqueeze(2)
        th = th.unsqueeze(2)
        tconf = tconf.unsqueeze(2)
        
        labels=t.cat((tx,ty,tw,th,tconf,tcls),2)
        
        return labels
        
if __name__ == '__main__':
    
    # DataLoader
    '''
    dataloader = t.utils.data.DataLoader(COCODataset("/home/lsk/Downloads/YOLOv3_PyTorch/data/coco/trainvalno5k.txt",
                                         (416, 416), is_training=True),
                                         batch_size=2, shuffle=True, num_workers=32, pin_memory=True)
    '''
    cfg=Configer("../cfg/yolov3.cfg")
    
    model = Darknet(cfg)
    #model.load_weights("../yolov3.weights")
    
    model.train(True)
    #model = nn.DataParallel(model)
    if t.cuda.is_available():        
        model.cuda()        
        
    dataloader = DataLoader(cfg)
    train_loader = dataloader.get_loader()
    loss_function = YOLO3Loss(cfg)

    # Start the training loop
    
    for epoch in range(1):
        for step, (images, img_size, gt_bboxes, gt_labels) in enumerate(train_loader):
            images=images.cuda()
            gt_labels=gt_labels
            gt_bboxes=gt_bboxes
            prediction = model(images)
            loss=loss_function(prediction, gt_labels, gt_bboxes)
            if step==0:
                break
    
    loss=loss.cpu().numpy()
    t.cuda.empty_cache()
    print(loss)
