import cv2
import numpy as np
import operator



import os
import re

import torch
import torch.nn as nn
import torchvision


import pandas as pd
import csv
import os 
import pickle

from collections import defaultdict

import Config.configuration as cfg


class DataProcess(object):
    def __init__(self, data_folder):
        self.data_folder = data_folder 


    def dump_file(self, data_list, file_type, split_folder="pcb_list"):
        save_path = os.path.join(cfg.data_path, split_folder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_file_path = os.path.join(save_path, file_type+'.pkl')
        with open(save_file_path, 'wb') as f:
            pickle.dump(data_list, f)

    def read_data_file(self):
        file_path = os.path.join(self.data_folder, cfg.file_name)
        df = pd.read_csv(file_path)
        pcb_imgs = [x for x in df['imageFile']]
        print("pcb image columns: ",len(pcb_imgs))
        pcb_ids = []
        pcb_container = {}
        traverse_ids = []
        pcb_comp_img = {}
        for img_name in pcb_imgs:
            indices = [i.start() for i in re.finditer('_', img_name)]
            pcb_id = img_name[0:indices[1]]

            # comp images from this pcb
            if pcb_id not in pcb_comp_img.keys():
                pcb_comp_img[pcb_id] = {'count':0}
            if img_name not in pcb_comp_img[pcb_id].keys():
                pcb_comp_img[pcb_id][img_name] = 0
                
            pcb_comp_img[pcb_id][img_name] += 1
            pcb_comp_img[pcb_id]['count'] += 1


            if img_name in traverse_ids:
                continue

            traverse_ids.append(img_name)
            #print("img name: ",img_name)
            
            pcb_ids.append(pcb_id)

            # different pcb imgs for a pcb_id
            if pcb_id not in pcb_container.keys():
                pcb_container[pcb_id] = {'count':0, 'img_name':[]}
            pcb_container[pcb_id]['count'] += 1
            pcb_container[pcb_id]['img_name'].append(img_name)

            #print("pcb id: ",img_name[0:indices[1]])
            #exit()
        pcb_ids = list(set(pcb_ids))
        #print("pcb ids: ",len(pcb_ids), pcb_ids)
        print("pcb ids: ",len(list(pcb_container.keys())))
        #pcb_container = dict( sorted(pcb_container.items(), key=operator.itemgetter(1),reverse=True))
        pcb_container = dict(sorted(pcb_container.items(), key=lambda x: x[1]['count'], reverse=True))
        #outcome = [(key, val) for key, val in pcb_container.items() if val>2]
        #print("pcb img used: ", outcome, len(outcome))
        #print("component imgs: ",len(list(pcb_comp_img.keys())))

        f = open(os.path.join('./output_check','pcb_stat.txt'), "w")
        for pcb_id, val in pcb_container.items():
            dict_to_write = {"id":pcb_id, "pcb_img":val['count'], "total_comp_img":pcb_comp_img[pcb_id]['count']}
            f.write(''+repr(dict_to_write) + '\n')
            dict_to_write = {"comp_img":pcb_comp_img[pcb_id]}
            f.write(''+repr(dict_to_write) + '\n')
            dict_to_write = {"names":val['img_name']}
            f.write(''+repr(dict_to_write) + '\n')
        f.close()


        result_holder = defaultdict(list)
        test_count = 0
        val_count = 0
        train_count = 0

        train_pcb_imgs = []
        val_pcb_imgs = []
        test_pcb_imgs = []

        for pcb_id, val in pcb_container.items():
            if val['count'] <= 2:
                result_holder[pcb_id].extend([(key, val) for key, val in pcb_comp_img[pcb_id].items() if 'count' not in key and "augmented" not in key])
            if val['count'] > 2:
                #train_pcb_imgs.extend([key for key, val in pcb_comp_img[pcb_id].items() if 'count' not in key])
                for pcb_img_name, n_comp_imgs in pcb_comp_img[pcb_id].items():
                    if 'count' not in pcb_img_name:
                        train_pcb_imgs.append(pcb_img_name)
                        train_count += n_comp_imgs



        #f = open(os.path.join('./output_check','pcb_2_img_stat.txt'), "w")
        #dict_to_write = {"id":pcb_id, "img_stats":img_stats}
            #f.write(''+repr(dict_to_write) + '\n')
        #f.close()
        
        print("train_imgs on PCB_Image > 2: ",train_count)

        for index, (pcb_id, img_stats) in enumerate(result_holder.items()):
            #previous 150-209 --> val
            for (img_name, val) in img_stats:
                if index < 130:
                    train_count += val
                    train_pcb_imgs.append(img_name)
                elif index >= 130 and index < 180:
                    val_count += val
                    val_pcb_imgs.append(img_name)
                elif index >= 180:
                    test_count += val
                    test_pcb_imgs.append(img_name)
        
                
            
        print("test_images: ",test_count)
        print("val images: ",val_count)
        print("train images: ",train_count)

        print(f'train_pcb:{len(train_pcb_imgs)} val_pcb_img:{len(val_pcb_imgs)} test_pcb_img:{len(test_pcb_imgs)}')
        
        self.dump_file(train_pcb_imgs, "train_pcb")
        self.dump_file(val_pcb_imgs, "val_pcb")
        self.dump_file(test_pcb_imgs, "test_pcb")
        return df

        
    def image_file_list(self, data_frame):   
        image_files = [x for x in data_frame['compImageFile']]
        return image_files

    def create_split(self, name, df):
        pcb_imgs = pickle.load(open(os.path.join(cfg.data_path, "pcb_list", name+'.pkl'), 'rb'))
        output = []
        for img in pcb_imgs:
            files = df.loc[df['imageFile'] == img]
            comp_images = self.image_file_list(files)
            output.extend(comp_images)
        return output




if __name__ == '__main__':
    data_process = DataProcess(cfg.data_folder)
    df = data_process.read_data_file()
    train_comps = data_process.create_split('train_pcb', df)
    val_comps = data_process.create_split('val_pcb', df)
    test_comps = data_process.create_split('test_pcb', df)
    print(f'train_comps:{len(train_comps)} val_comps:{len(val_comps)} test_comps:{len(test_comps)}')
    data_process.dump_file(train_comps, "train", "split")
    data_process.dump_file(val_comps, "val", "split")
    data_process.dump_file(test_comps, "test", "split")






