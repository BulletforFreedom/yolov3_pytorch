# -*- coding: utf-8 -*-

class Configer(object):
    def __init__(self, cfgfile):
        self.cfgfile=cfgfile
        self.blocks = []
        self.net_info=None
        self.parse_cfg()   
    
    def get_net_info(self):
        return self.net_info
    
    def get_blocks(self):
        return self.blocks
    
    def get_inp_dim(self):
        return int(self.net_info['height'])
    
    def get_num_classes(self):
        return int(self.net_info['classes'])
    
    def get_loss_lambda(self):
        return [float(self.net_info['coord_scale']), float(self.net_info['object_scale']),
                float(self.net_info['noobject_scale']), float(self.net_info['class_scale'])]
    
    def get_iou_threshold(self):
        return float(self.net_info['ignore_iou_thresh'])
    
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
