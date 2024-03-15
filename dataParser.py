import os
import numpy as np
from multiprocessing import Pool
from functools import partial
import pickle
import random
import re
import copy

import cv2
import pickle
from sklearn.model_selection import train_test_split

import statistics
from datetime import datetime

import sys

import Config.configuration as cfg
import Config.parameters as params


class PCBParser(object):
    def __init__(self, data_path, run_id, val=False, test=False):

        super(PCBParser,self).__init__()

        self.data_path = data_path  
        self.img_files = []
        self.aug_img = 0
        self.test = test
        self.val = val
        if not val:
            if not test:
                name = 'train.pkl'
            else:
                name = 'test.pkl'
        else:
            name = 'val.pkl'

        img_file_names = pickle.load(open(os.path.join(self.data_path, name), 'rb'))
        # print(img_file_names[:10])
        # exit()
        #random.shuffle(img_file_names)

        self.img_files = self._load_data(img_file_names, isDictionary=True)
        if not self.val and not self.test and save_selected_bbox_list:
            save_file_path = os.path.join(cfg.data_path, run_id)
            if not os.path.exists(save_file_path):
                os.makedirs(save_file_path)
            save_file_path = os.path.join(save_file_path, 'train_boxed.pkl')
            with open(save_file_path, 'wb') as f:
                pickle.dump(self.img_files, f)
                print("bboxed train list saved")
            random.shuffle(self.img_files)

    def __len__(self):
        return len(self.img_files)
 

    def _stats(self, img_name):
        if 'augmented' in img_name:
            self.aug_img += 1
        

    def _load_data(self, img_file_names, isDictionary=False):
        data = []
        # path = os.path.join(cfg.data_folder, split)
        # path = cfg.data_folder      #data_folder needs to be changed for new split
        counter = 0
        bbox_annot = 0
        holder = []

        if cfg.split_set =="filtered_set" and isDictionary and not self.val and not self.test:
            # consider those images where p<3 and a<=0.81
            # Then there's a significant difference between actual and bbox annotation
            bbox_set = [p for p in img_file_names if p['perimeter']<3 and p['area']<=0.81]
            bbox_number = cfg.bbox_number - 275  # 275 hard set always included    
            new_bbox_set = bbox_set[:bbox_number] 

        for img_info in img_file_names:
            img_dict = {'bbox':0}
            if not self.val and not self.test and params.allow_bbox:
                if isDictionary:
                    img_path = img_info['img']
                    name = img_path[img_path.rindex('/')+1:]
                else:
                    name = img_info
                indices = [match.start() for match in re.finditer(r'_', name)]
                pcb_id = name[0:indices[1]]
                if isDictionary and cfg.split_set=="filtered_set":
                    if img_info['perimeter'] >= 3:      # 275 hard set
                        img_dict['bbox'] = 1
                        bbox_annot += 1
                    else:                   # otherwise only take certain % from
                                            # bbox set
                        if img_info in new_bbox_set:
                            img_dict['bbox'] = 1
                            bbox_annot += 1
                        # elif img_info in bbox_set:
                        #     continue
            if not isDictionary:
                img_dict['img'] = os.path.join(path, "images", name)
                img_dict['mask'] = os.path.join(path, "label_masks", name)
            else:
                img_dict['img'] = img_info['img']
                img_dict['mask'] = img_info['mask']
                img_dict['iou_bbox'] = img_info['iou_bbox']
            data.append(copy.deepcopy(img_dict))
            counter += 1
        print(f'total:{counter}, bboxes:{bbox_annot}')
        #print('selected box ids: ',list(set(holder)))
        return data




if __name__=="__main__":
    run_started = datetime.today().strftime('%m-%d-%y_%H%M')
    train_parser = PCBParser(cfg.data_path, run_id=run_started)
    val_parser = PCBParser(cfg.data_path, run_id=run_started, val=True)
    test_parser = PCBParser(cfg.data_path, run_id=run_started, test=True)
    ##print(parser.classes_to_idx)
    #print(len(parser.img_files))
    ##print(parser.img_files[100])