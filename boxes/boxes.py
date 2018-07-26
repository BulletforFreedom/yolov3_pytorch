# -*- coding: utf-8 -*-
import torch

class Boxes(object):    
    
    @staticmethod
    def bbox_iou(box1,boxn,x1y1x2y2=True):
        """
        Returns the IoU of two bounding boxes
        """
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = boxn[:, 0] - boxn[:, 2] / 2, boxn[:, 0] + boxn[:, 2] / 2
            b2_y1, b2_y2 = boxn[:, 1] - boxn[:, 3] / 2, boxn[:, 1] + boxn[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
            b2_x1, b2_y1, b2_x2, b2_y2 = boxn[:,0], boxn[:,1], boxn[:,2], boxn[:,3]
    
        # get the corrdinates of the intersection rectangle
        inter_rect_x1 =  torch.max(b1_x1, b2_x1)
        inter_rect_y1 =  torch.max(b1_y1, b2_y1)
        inter_rect_x2 =  torch.min(b1_x2, b2_x2)
        inter_rect_y2 =  torch.min(b1_y2, b2_y2)
        # Intersection area
        inter_area =    torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                        torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    
        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    
        return iou
    
    @staticmethod
    def get_nms(image_pred_class,nms_conf):
        i=0
        while i < image_pred_class.size(0)-1:
            
            ious=Boxes.bbox_iou(image_pred_class[i].unsqueeze(0),image_pred_class[i+1:])
            #Zero out all the detections that have IoU > treshhold
            iou_mask = (ious < nms_conf).float().unsqueeze(1)
            image_pred_class[i+1:] *= iou_mask       
            
            #Remove the non-zero entries
            non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
            image_pred_class = image_pred_class[non_zero_ind]
            try:
                image_pred_class.size(1)
            except:
                return image_pred_class.unsqueeze(0)
            i+=1
        return image_pred_class