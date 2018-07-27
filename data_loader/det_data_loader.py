#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
from torch.utils import data

import data_augmentation.aug_transforms as aug_trans
import data_loader.transforms as trans
#from datasets.det.ssd_data_loader import SSDDataLoader
from data_loader.yolo_data_loader import YoloDataLoader
from logger import Logger as Log


class DetDataLoader(object):

    def __init__(self, configer, is_debug=False):
        self.configer = configer
        self.is_debug = is_debug

        #self.aug_train_transform = aug_trans.AugCompose(self.configer, split='train')

        #self.aug_val_transform = aug_trans.AugCompose(self.configer, split='val')
        
        self.img_transform = trans.Compose()
        self.img_transform.add(trans.ResizeImage(self.configer.get_inp_dim()))
        self.img_transform.add(trans.ToTensor())
        #self.img_transform.add(trans.Normalize(self.configer.get_dataset_mean(),
                                               #self.configer.get_dataset_std()))
        
    def get_loader(self):
        if self.configer.get_method() == 'single_shot_detector':
            '''
            trainloader = data.DataLoader(
                SSDDataLoader(root_dir=os.path.join(self.configer.get('data', 'data_dir'), 'train'),
                              aug_transform=self.aug_train_transform,
                              img_transform=self.img_transform,
                              configer=self.configer),
                batch_size=self.configer.get('data', 'train_batch_size'), shuffle=True,
                num_workers=self.configer.get('data', 'workers'), collate_fn=self._detection_collate, pin_memory=True)

            return trainloader
            '''
        elif self.configer.get_method() == 'faster_rcnn':
            '''
            trainloader = data.DataLoader(
                FRDataLoader(root_dir=os.path.join(self.configer.get('data', 'data_dir'), 'train'),
                             aug_transform=self.aug_train_transform,
                             img_transform=self.img_transform,
                             configer=self.configer),
                batch_size=self.configer.get('data', 'train_batch_size'), shuffle=True,
                num_workers=self.configer.get('data', 'workers'), collate_fn=self._detection_collate, pin_memory=True)

            return trainloader
            '''
        elif self.configer.get_method() == 'yolov3':
            if self.configer.is_train():
                bs=self.configer.get_batch_size()
                train_info='train'
                #aug_transform=self.aug_train_transform
                shuffle=True
            else:
                bs=1
                train_info='val'
                #aug_transform=self.aug_val_transform
                shuffle=False
                
            if self.is_debug:
                detection_collate = self._detection_collate_debug
            else:
                detection_collate=self._detection_collate
                
            trainloader = data.DataLoader(
                YoloDataLoader(root_dir=os.path.join(self.configer.get_data_dir(), train_info),
                             aug_transform=None,
                             img_transform=self.img_transform,
                             configer=self.configer),
                batch_size=bs, shuffle=shuffle,
                num_workers=self.configer.get_num_workers(), 
                collate_fn=detection_collate, pin_memory=True)
            return trainloader

        else:
            Log.error('Method: {} loader is invalid.'.format(self.configer.get('method')))
            return None

    def get_valloader(self):
        if self.configer.get('method') == 'single_shot_detector':
            '''
            valloader = data.DataLoader(
                SSDDataLoader(root_dir=os.path.join(self.configer.get('data', 'data_dir'), 'val'),
                              aug_transform=self.aug_val_transform,
                              img_transform=self.img_transform,
                              configer=self.configer),
                batch_size=self.configer.get('data', 'val_batch_size'), shuffle=False,
                num_workers=self.configer.get('data', 'workers'), collate_fn=self._detection_collate, pin_memory=True)

            return valloader
            '''

        elif self.configer.get('method') == 'faster_rcnn':
            '''
            valloader = data.DataLoader(
                FRDataLoader(root_dir=os.path.join(self.configer.get('data', 'data_dir'), 'val'),
                             aug_transform=self.aug_val_transform,
                             img_transform=self.img_transform,
                             configer=self.configer),
                batch_size=self.configer.get('data', 'val_batch_size'), shuffle=False,
                num_workers=self.configer.get('data', 'workers'), collate_fn=self._detection_collate, pin_memory=True)

            return valloader
            '''

        else:
            Log.error('Method: {} loader is invalid.'.format(self.configer.get('method')))
            return None

    @staticmethod
    def _detection_collate(batch):
        """Custom collate fn for dealing with batches of images that have a different
            number of associated object annotations (bounding boxes).
            Arguments:
                batch: (tuple) A tuple of tensor images and lists of annotations
            Return:
                A tuple containing:
                    1) (tensor) batch of images stacked on their 0 dim
                    2) (list of tensors) annotations for a given image are stacked on
                                         0 dim
            """
        imgs = []
        bboxes = []
        labels = []
        for sample in batch:
            imgs.append(sample[0])
            bboxes.append(sample[2])
            labels.append(sample[3])
            

        return torch.stack(imgs, 0), bboxes, labels

    @staticmethod
    def _detection_collate_debug(batch):
        """Custom collate fn for dealing with batches of images that have a different
            number of associated object annotations (bounding boxes).
            Arguments:
                batch: (tuple) A tuple of tensor images and lists of annotations
            Return:
                A tuple containing:
                    1) (tensor) batch of images stacked on their 0 dim
                    2) (list of tensors) annotations for a given image are stacked on
                                         0 dim
            """
        imgs = []
        bboxes = []
        labels = []
        img_dir_list = []
        for sample in batch:
            imgs.append(sample[0])
            img_dir_list.append(sample[1])
            bboxes.append(sample[2])
            labels.append(sample[3])
            

        return torch.stack(imgs, 0), img_dir_list, bboxes, labels
