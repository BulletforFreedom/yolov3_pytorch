# -*- coding: utf-8 -*-
from data.data_info import coco2014 as data_info

class Configer(object):
    def __init__(self, cfgfile):
        self.cfgfile=cfgfile
        self.blocks = []
        self.net_info=None
        self.parse_cfg()
        #self.net_info['dataset']=data_info
        self.net_info['strides'] = -1
        self.net_info['scaled_anchor_list'] = []
        self.net_info['anchor_list'] = []
        
    def set_strides(self, strides):
        self.net_info['strides'] = strides
        
    def set_scaled_anchor_list(self, scaled_anchor_list):
        self.net_info['scaled_anchor_list'].append(scaled_anchor_list)
        
    def set_anchor_list(self, anchor):
        self.net_info['anchor_list'].append(anchor)
    
    def get_net_info(self):
        return self.net_info
    
    def is_train(self):
        return self.net_info['train']
    
    def get_blocks(self):
        return self.blocks
    
    def get_inp_dim(self):
        return int(self.net_info['inp_dim'])
    
    def get_num_classes(self):
        return int(self.net_info['classes'])
    
    def get_loss_lambda(self):
        return [float(self.net_info['coord_scale']), float(self.net_info['object_scale']),
                float(self.net_info['noobject_scale']), float(self.net_info['class_scale'])]
    
    def get_iou_threshold(self):
        return float(self.net_info['ignore_iou_thresh'])
    
    def get_strides(self):
        return self.net_info['strides']
    
    def get_anchors(self):
        return self.net_info['anchors']
    
    def get_anchor_list(self):
        return self.net_info['anchor_list']
        
    def get_scaled_anchor_list(self):
        return self.net_info['scaled_anchor_list']
    
    def get_method():
        return data_info['method']
    
    def get_batch_size():
        return data_info['batch_size']
    
    def get_dataset_mean():
        return data_info['mean']
    
    def get_dataset_std():
        return data_info['std']
    
    def get_dataset_name_seq():
        return data_info['name_seq']
    
    def get_data_dir():
        return data_info['data_dir']
    
    def parse_cfg(self):
        """
        Takes a configuration file
        
        Returns a list of blocks. Each blocks describes a block in the neural
        network to be built. Block is represented as a dictionary in the list
        
        """
        file = open(self.cfgfile, 'r')
        lines = file.read().split('\n')     #store the lines in a list
        lines = [x for x in lines if len(x) > 0] #get read of the empty lines 
        lines = [x for x in lines if x[0] != '#']  
        lines = [x.strip() for x in lines]
    
        
        block = {}
        
        
        for line in lines:
            if line[0] == "[":               #This marks the start of a new block
                if len(block) != 0:
                    self.blocks.append(block)
                    block = {}
                block["type"] = line[1:-1].rstrip()                
            else:
                key,value = line.split("=")
                    
                block[key.rstrip()] = value.lstrip()
        self.blocks.append(block)
        self.net_info=self.blocks.pop(0)

if __name__ == '__main__':
    cfg=Configer("../yolov3_train.cfg")
    net_info=cfg.get_net_info()