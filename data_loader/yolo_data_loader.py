#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Faster R-CNN data loader


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
#from PIL import Image
import cv2
import numpy as np
import torch
import torch.utils.data as data
from data_loader.json_helper import JsonHelper

#from datasets.det.det_data_utilizer import DetDataUtilizer
from data_loader.det_transforms import ResizeBoxes
#from utils.helpers.image_helper import ImageHelper
from logger import Logger as Log


class YoloDataLoader(data.Dataset):

    def __init__(self, root_dir=None, aug_transform=None,
                 img_transform=None, configer=None):
        super(YoloDataLoader, self).__init__()
        self.img_list, self.json_list = self.__list_dirs(root_dir)
        self.configer = configer
        self.aug_transform = aug_transform
        self.img_transform = img_transform
        #self.det_data_utilizer = DetDataUtilizer(configer)
    
    def __getitem__(self, index):
        #img = Image.open(self.img_list[index]).convert('RGB')
        img_dir=self.img_list[index]
        img = cv2.imread(self.img_list[index])
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[:,:,:]/255.0
        labels, bboxes = self.__read_json_file(self.json_list[index])

        if self.aug_transform is not None:
            img, bboxes = self.aug_transform(img, bboxes=bboxes)

        #img, bboxes, labels = ResizeBoxes()(img, bboxes, labels)
        labels = torch.from_numpy(np.array(labels)).long()
        bboxes = torch.from_numpy(np.array(bboxes)).float()

        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, img_dir, bboxes, labels
    

    def __len__(self):

        return len(self.img_list)

    def __read_json_file(self, json_file):
        """
            filename: JSON file

            return: three list: key_points list, centers list and scales list.
        """
        json_dict = JsonHelper.load_file(json_file)

        #img_size = [json_dict['width'], json_dict['height']]
        
        labels = list()
        bboxes = list()

        for object in json_dict['objects']:
            labels.append(object['label'])
            bboxes.append(object['bbox'])

        return labels, bboxes

    def __list_dirs(self, root_dir):
        img_list = list()
        json_list = list()
        image_dir = os.path.join(root_dir, 'image')
        json_dir = os.path.join(root_dir, 'json')

        img_extension = os.listdir(image_dir)[0].split('.')[-1]

        for file_name in os.listdir(json_dir):
            image_name = '.'.join(file_name.split('.')[:-1])
            img_list.append(os.path.join(image_dir, '{}.{}'.format(image_name, img_extension)))
            json_path = os.path.join(json_dir, file_name)
            json_list.append(json_path)
            if not os.path.exists(json_path):
                Log.error('Json Path: {} not exists.'.format(json_path))
                exit(1)

        return img_list, json_list