import os
import sys
import glob
from PIL import Image
import argparse
import json
import matplotlib.pyplot as plt
import skimage.io as io
import cv2
from labelme import utils
import numpy as np
import sys


class transform():
    """docstring for transform"""

    def __init__(self, is_gt, src_file, tar_file, images_file=None, src_gt=None, mode=None):
        # Mode list: 'txt_to_voc','txt_to_coco','voc_to_coco','voc_to_csv'
        #Parameters:  is_gt: flag for gt_type or pred_type
        #             src_file: dictionary of src_file
        #             tar_file: dictionary of output_pos
        #             image_file: dictionary of images
        #             src_gt: if transform pred_voc to pred_coco, gt_category map has to be pointed
        

        self.is_gt = is_gt
        self.src_file = src_file
        self.tar_file = tar_file
        self.images_file = images_file
        self.src_gt = src_gt
        self.mode = mode
    
            # print(self.map_dic[])

        if mode == 'txt_to_voc':
            self.txt_to_voc()
        if mode == 'voc_to_coco':
            if is_gt == False:
                with open(src_gt) as f:
                    js = json.load(f)
                    self.map_dic = js['categories']
            xml_file = glob.glob(self.src_file + '/*.xml')
            PascalVOC2coco(xml_file, self.tar_file,
                           is_gt=self.is_gt, map_dic=self.map_dic)

    def txt_to_voc(self):
        # VEDAI 图像存储位置
        src_img_dir = self.images_file
        # VEDAI 图像的 ground truth 的 txt 文件存放位置
        src_txt_dir = self.src_file
        src_xml_dir = self.tar_file

        img_Lists = glob.glob(src_img_dir + '/*.jpg')
        img_basenames = []  # e.g. 100.jpg
        for item in img_Lists:
            img_basenames.append(os.path.basename(item))

        img_names = []  # e.g. 100
        for item in img_basenames:
            temp1, temp2 = os.path.splitext(item)
            img_names.append(temp1)

        for img in img_names:
            im = Image.open((src_img_dir + '/' + img + '.jpg'))
            width, height = im.size

            # open the crospronding txt file
            gt = open(src_txt_dir + '/' + img + '.txt').read().splitlines()
            #gt = open(src_txt_dir + '/gt_' + img + '.txt').read().splitlines()

            # write in xml file
            #os.mknod(src_xml_dir + '/' + img + '.xml')
            xml_file = open((src_xml_dir + '/' + img + '.xml'), 'w')
            xml_file.write('<annotation>\n')
            xml_file.write('    <folder>VOC2007</folder>\n')
            xml_file.write('    <filename>' + str(img) +
                           '.jpg' + '</filename>\n')
            xml_file.write('    <size>\n')
            xml_file.write('        <width>' + str(width) + '</width>\n')
            xml_file.write('        <height>' + str(height) + '</height>\n')
            xml_file.write('        <depth>3</depth>\n')
            xml_file.write('    </size>\n')

            # write the region of image on xml file
            for img_each_label in gt:
                # 这里如果txt里面是以逗号‘，’隔开的，那么就改为spt = img_each_label.split(',')。
                spt = img_each_label.split(' ')
                if self.is_gt:
                    wid = float(spt[3]) * width
                    hei = float(spt[4]) * height
                    cen_x = float(spt[1]) * width
                    cen_y = float(spt[2]) * height
                else:
                    wid = float(spt[4]) * width
                    hei = float(spt[5]) * height
                    cen_x = float(spt[2]) * width
                    cen_y = float(spt[3]) * height
                x1 = round(cen_x - wid / 2)
                x2 = round(cen_x + wid / 2)
                y1 = round(cen_y - hei / 2)
                y2 = round(cen_y + wid / 2)
                xml_file.write('    <object>\n')
                xml_file.write('        <name>' + str(spt[0]) + '</name>\n')
                xml_file.write('        <pose>Unspecified</pose>\n')
                xml_file.write('        <truncated>0</truncated>\n')
                xml_file.write('        <difficult>0</difficult>\n')
                if self.is_gt == False:
                    xml_file.write('        <score>' +
                                   str(spt[1]) + '</score>\n')
                xml_file.write('        <bndbox>\n')
                xml_file.write('            <xmin>' + str(x1) + '</xmin>\n')
                xml_file.write('            <ymin>' + str(y1) + '</ymin>\n')
                xml_file.write('            <xmax>' + str(x2) + '</xmax>\n')
                xml_file.write('            <ymax>' + str(y2) + '</ymax>\n')
                xml_file.write('        </bndbox>\n')
                xml_file.write('    </object>\n')

            xml_file.write('</annotation>')


class PascalVOC2coco(object):
    def __init__(self, xml=[], save_json_path='./new.json', is_gt=True, map_dic=None):
        '''
        :param xml: 所有Pascal VOC的xml文件路径组成的列表
        :param save_json_path: json保存位置
        '''
        self.xml = xml
        self.save_json_path = save_json_path
        self.images = []
        self.categories = []
        self.annotations = []
        self.area = 0.0
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0
        self.score = 0
        self.is_gt = is_gt
        self.map_dic = map_dic
        self.transpose = []
        self.save_json()

    def data_transfer(self):
        for num, json_file in enumerate(self.xml):

            # 进度输出
            sys.stdout.write('\r>> Converting image %d/%d' % (
                num + 1, len(self.xml)))
            sys.stdout.flush()

            self.json_file = json_file
            # print(json_file)
            self.num = num
            path = os.path.dirname(self.json_file)
            path = os.path.dirname(path)
            # path=os.path.split(self.json_file)[0]
            # path=os.path.split(path)[0]
            obj_path = glob.glob(os.path.join(
                path, 'SegmentationObject', '*.png'))
            with open(json_file, 'r') as fp:
                for p in fp:
                    # if 'folder' in p:
                    #     folder =p.split('>')[1].split('<')[0]
                    if 'filename' in p:
                        self.filen_ame = p.split('>')[1].split('<')[0]
                        self.path = os.path.join(
                            path, 'SegmentationObject', self.filen_ame.split('.')[0] + '.png')
                        # if self.path not in obj_path:
                        #     break

                    if 'width' in p:
                        self.width = int(p.split('>')[1].split('<')[0])
                    if 'height' in p:
                        self.height = int(p.split('>')[1].split('<')[0])

                        self.images.append(self.image())

                    if '<object>' in p:
                        # 类别
                        d = [next(fp).split('>')[1].split('<')[0]
                             for _ in range(10)]
                        self.supercategory = d[0]
                        if self.supercategory not in self.label:
                            self.categories.append(self.categorie())
                            self.label.append(self.supercategory)
                        # print(d)
                        # return 0
                        # 边界框
                        # print(d)
                        if self.is_gt == False:
                            self.score = float(d[-6])
                        x1 = int(d[-4])
                        y1 = int(d[-3])
                        x2 = int(d[-2])
                        y2 = int(d[-1])
                        self.rectangle = [x1, y1, x2, y2]
                        # COCO 对应格式[x,y,w,h]
                        self.bbox = [x1, y1, x2 - x1, y2 - y1]

                        self.annotations.append(self.annotation())
                        self.annID += 1

        sys.stdout.write('\n')
        sys.stdout.flush()

    def image(self):
        image = {}
        image['height'] = self.height
        image['width'] = self.width
        image['id'] = self.num + 1
        image['file_name'] = self.filen_ame
        return image

    def categorie(self):
        categorie = {}
        categorie['supercategory'] = self.supercategory
        categorie['id'] = len(self.label) + 1  # 0 默认为背景
        categorie['name'] = self.supercategory
        return categorie

    def annotation(self):
        annotation = {}
        # annotation['segmentation'] = [self.getsegmentation()]
        annotation['segmentation'] = [list(map(float, self.getsegmentation()))]
        annotation['iscrowd'] = 0
        annotation['image_id'] = self.num + 1
        # annotation['bbox'] = list(map(float, self.bbox))
        annotation['bbox'] = self.bbox
        annotation['category_id'] = self.getcatid(self.supercategory)
        annotation['id'] = self.annID
        annotation['area'] = self.area
        if self.is_gt == False:
            annotation['score'] = self.score
        return annotation

    def getcatid(self, label):
        for categorie in self.categories:
            if label == categorie['name']:
                return categorie['id']
        return -1

    def getsegmentation(self):

        try:
            mask_1 = cv2.imread(self.path, 0)
            mask = np.zeros_like(mask_1, np.uint8)
            rectangle = self.rectangle
            mask[rectangle[1]:rectangle[3], rectangle[0]:rectangle[2]] = mask_1[rectangle[1]:rectangle[3],
                                                                                rectangle[0]:rectangle[2]]

            # 计算矩形中点像素值
            mean_x = (rectangle[0] + rectangle[2]) // 2
            mean_y = (rectangle[1] + rectangle[3]) // 2

            end = min((mask.shape[1], int(rectangle[2]) + 1))
            start = max((0, int(rectangle[0]) - 1))

            flag = True
            for i in range(mean_x, end):
                x_ = i
                y_ = mean_y
                pixels = mask_1[y_, x_]
                if pixels != 0 and pixels != 220:  # 0 对应背景 220对应边界线
                    mask = (mask == pixels).astype(np.uint8)
                    flag = False
                    break
            if flag:
                for i in range(mean_x, start, -1):
                    x_ = i
                    y_ = mean_y
                    pixels = mask_1[y_, x_]
                    if pixels != 0 and pixels != 220:
                        mask = (mask == pixels).astype(np.uint8)
                        break
            self.mask = mask

            return self.mask2polygons()

        except:
            return [0]

    def mask2polygons(self):
        '''从mask提取边界点'''
        contours = cv2.findContours(
            self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找到轮廓线
        bbox = []
        for cont in contours[1]:
            [bbox.append(i) for i in list(cont.flatten())]
            # map(bbox.append,list(cont.flatten()))
        return bbox  # list(contours[1][0].flatten())

    # '''
    def getbbox(self, points):
        '''边界点生成mask，从mask提取定位框'''
        # img = np.zeros([self.height,self.width],np.uint8)
        # cv2.polylines(img, [np.asarray(points)], True, 1, lineType=cv2.LINE_AA)  # 画边界线
        # cv2.fillPoly(img, [np.asarray(points)], 1)  # 画多边形 内部像素值为1
        polygons = points
        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):
        '''从mask反算出其边框
        mask：[h,w]  0、1组成的图片
        1对应对象，只需计算1对应的行列号（左上角行列号，右下角行列号，就可以算出其边框）
        '''
        # np.where(mask==1)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]
        # 解析左上角行列号
        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        # 解析右下角行列号
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        # return [(left_top_r,left_top_c),(right_bottom_r,right_bottom_c)]
        # return [(left_top_c, left_top_r), (right_bottom_c, right_bottom_r)]
        # return [left_top_c, left_top_r, right_bottom_c, right_bottom_r]  # [x1,y1,x2,y2]
        return [left_top_c, left_top_r, right_bottom_c - left_top_c,
                right_bottom_r - left_top_r]  # [x1,y1,w,h] 对应COCO的bbox格式

    def polygons_to_mask(self, img_shape, polygons):
        '''边界点生成mask'''
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    # '''
    def data2coco(self):
        data_coco = {}
        data_coco['images'] = self.images
        data_coco['categories'] = self.categories
        data_coco['annotations'] = self.annotations
        return data_coco

    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
        self.transpose = self.data_coco['categories']
        # print(self.transpose)
        print(self.map_dic)
        # 保存json文件
        if self.is_gt == False:
            for item in self.data_coco['annotations']:
                item['category_id'] = self.transpose[item['category_id'] -
                                                     1]['supercategory']
                for i in self.map_dic:
                    if i['supercategory'] == item['category_id']:
                        item['category_id'] = i['id']
                        break
                        # print(item['category_id'])
                        # print(i['id'])
                # print(item)
        json.dump(self.data_coco, open(self.save_json_path, 'w'),
                  indent=4)  # indent=4 更加美观显示


# xml_train = glob.glob('./VOC_dataset/train/Annotations/*.xml')
# # xml_file = ['./Annotations/frames_00032.xml']
# xml_test = glob.glob('./VOC_dataset/test/Annotations/*.xml')
# xml_val = glob.glob('./VOC_dataset/val/Annotations/*.xml')

# PascalVOC2coco(xml_train, './COCO_dataset/train/train.json')
# PascalVOC2coco(xml_test, './COCO_dataset/test/test.json')
# PascalVOC2coco(xml_val, './COCO_dataset/val/val.json')

# xml_file = ['test/pred_voc_annotations/1.xml']
# PascalVOC2coco(xml_file, './test/pred_coco_annotations/1.json')


if __name__ == '__main__':
    a = transform(is_gt=False, src_file='test/pred_voc_annotations',
                  images_file='test/image', tar_file='test/pred_coco', src_gt='test/gt_coco_annotations/1.json', mode='voc_to_coco')
