import cv2
import numpy as np


import os


import torch
import torch.nn as nn
import torchvision


import pandas as pd
import csv
import os 
import pickle


import Config.configuration as cfg


class DataProcess(object):
    def __init__(self, data_folder):
        self.data_folder = data_folder 

    def read_data_file(self):
        file_path = os.path.join(self.data_folder, cfg.file_name)
        df = pd.read_csv(file_path)

        train_files = df.loc[df['dataType'] == "train"]
        val_files = df.loc[df['dataType'] == "val"]
        test_files = df.loc[df['dataType'] == "test"]

        train_images = self.image_file_list(train_files)
        
        validation_images = self.image_file_list(val_files)
        test_images = self.image_file_list(test_files)
        print(len(train_images))
        print(len(validation_images))
        print(len(test_images))
        self.dump_file(train_images, 'train')
        self.dump_file(validation_images, 'val')
        self.dump_file(test_images, 'test')

    def dump_file(self, data_list, file_type):
        save_file_path = os.path.join(cfg.data_path, file_type+'.pkl')
        with open(save_file_path, 'wb') as f:
            pickle.dump(data_list, f)
        
    def image_file_list(self, data_frame):   
        image_files = [x for x in data_frame['compImageFile']]
        return image_files

if __name__ == '__main__':
    data_process = DataProcess(cfg.data_folder)
    data_process.read_data_file()




