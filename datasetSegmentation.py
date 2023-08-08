#Python Libraries
from collections import defaultdict
import numpy as np
from PIL import Image
import copy
import glob
import random
import sys
import os
from datetime import datetime
import time
from augmentation import Augmentation


#Torch Library
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch  

#TorchVision Library
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image


# Third-Party Libraries
import cv2
import xml.etree.ElementTree as ET
import re

# Project Files
from utils import convert_to_bbox_mask, bm1
from dataParser import PCBParser
from Config import configuration as cfg
from Config import parameters as params

from profiler import Profiler




class PCBDataset(Dataset):
    def __init__(self, img_files, transform = None, val=False):

        super(PCBDataset,self).__init__()

        self.img_files = img_files
        self.valid = val
        self.transform = transform
        self.count = 0

    def convert_to_tensor(self, img):
        img_torch = torch.from_numpy(img)
        img_torch = img_torch.type(torch.FloatTensor)
        img_torch = img_torch.permute(-1, 0, 1)
        return img_torch


    def __getitem__(self, index):
        #print(f'rank {torch.distributed.get_rank()} fetch sample {idx}')
        #profiler.tick("initiation")
        image_container =  self.img_files[index%len(self.img_files)]
        #print(image_container)
        #profiler.tick("img access")
        img_path = image_container['img']
        mask_path = image_container['mask']
        #print('img path: ',img_path)
        
        # img_id = img_path[img_path.rindex('/')+1:]
        # indexes = [match.start() for match in re.finditer(r'_', img_id)]
        # pcb_id = img_id[0:indexes[1]]

        img = cv2.imread(img_path)      # 512 x 512 x 3
        #profiler.tick("reading image completed")
        ##16-bit mask read
        mask16 = cv2.imread(mask_path, -1)
        mask = mask16.astype('uint8')   #mask is already between -> [0,1]
        mask = mask*255                 #mask is between -> [0,255]
        bbox_label = 0
        if not self.valid and self.transform is not None:
            img, mask = self.transform.generation(img, mask, params.isReduced)
            if image_container['bbox'] == 1:
                #self.count += 1
                bbox_label = 1
                #print('occurs')
                mask_bbox = convert_to_bbox_mask(copy.deepcopy(mask))
            else:
                mask_bbox = copy.deepcopy(mask)
                bbox_label = 0

        #Normalization and convert to tensor
        img = img/255.0
        img_tensor = self.convert_to_tensor(img)        # (3, 512, 512)

        mask = np.expand_dims(mask, axis=2)             # mask -> (512,512,1)
        mask = mask/255                                 # mask was converted between -> [0, 255].. now -> [0,1]
        mask_tensor = self.convert_to_tensor(mask)      # (1, 512, 512)
        mask_tensor = torch.squeeze(mask_tensor)
        mask_tensor = mask_tensor.type(torch.LongTensor)

        bbox_label = np.asarray(bbox_label)
        bbox_label = torch.from_numpy(bbox_label)

        if not self.valid:
            mask_bbox = np.expand_dims(mask_bbox, axis=2)             # mask -> (512,512,1)
            mask_bbox = mask_bbox/255                                 # mask was converted between -> [0, 255].. now -> [0,1]
            mask_bbox_tensor = self.convert_to_tensor(mask_bbox)      # (1, 512, 512)
            mask_bbox_tensor = torch.squeeze(mask_bbox_tensor)
            mask_bbox_tensor = mask_bbox_tensor.type(torch.LongTensor)
            return img_tensor, mask_tensor, mask_bbox_tensor, bbox_label
        else:
            return img_tensor, mask_tensor


    def __len__(self):
        return len(self.img_files)

if __name__ == "__main__":
    run_id = datetime.today().strftime('%m-%d-%y_%H%M')
    print('run id: ',run_id)
    

    # parser = PCBParser(cfg.data_path, run_started)
    augImg = Augmentation()
    val_parser = PCBParser(cfg.data_path, run_id, val=True)
    test_parser = PCBParser(cfg.data_path, run_id, test=True)
    train_parser = PCBParser(cfg.data_path, run_id)
    
    val_dataset = PCBDataset(val_parser.img_files, val=True)
    test_dataset = PCBDataset(test_parser.img_files, val=True)
    train_dataset = PCBDataset(train_parser.img_files, transform=augImg)

    #print('length val dataset: ',len(val_dataset))
    #print('length test dataset: ',len(test_dataset))
    #print('length train dataset: ',len(train_dataset))

    #exit()

    val_dataloader = DataLoader(val_dataset, batch_size=64, num_workers=4, pin_memory=True, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=4, pin_memory=True, shuffle=False)
    train_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=4, pin_memory=True, shuffle=False)
    #print("Train dataset length: ", len(train_dataset))
    #print('train augmented_image: ', parser.aug_img)
    #print('Train dataloader: ', len(train_dataloader))
    #print("initial bbox count: ",train_dataset.count)
    folder = -1
    root_path = os.path.join('./output_check', 'mask_output')
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    count = 0
    holder = {}
    low_holder = {}

    for i, (inputs, masks) in enumerate(val_dataloader):
        
        if i%1000==0:
            print('itr: ',i)
            continue
            #folder = i
            path = os.path.join(root_path, str(i))
            if not os.path.exists(path):
                os.makedirs(path)
        continue
        input_img_path = '%s/input_%05d.jpg' % (path, i)
        mask_path = '%s/mask_%05d.jpg' % (path, i)
        mask_bbox_path = '%s/mask_bbox_%05d.jpg' % (path, i)

        #torchvision.utils.save_image(inputs, input_img_path, normalize=True, nrow=1, range=(0, 1))
        #torchvision.utils.save_image(masks, mask_path, normalize=True, nrow=1, range=(0, 1))
        iou = iou.numpy()[0]
        #print("iou: ", iou)
        pcb_id = pcb_id[0]
        #exit()
        if pcb_id not in holder:
            holder[pcb_id] = {'high':0, 'low':0}
        if iou>=0.9:
            holder[pcb_id]['high']+=1
            count+=1
        else:
            holder[pcb_id]['low']+=1
        continue
        masks_bbox = torch.squeeze(masks_bbox)
        print(masks_bbox.size())
        masks_bbox = masks_bbox.permute(1, 2, 0)
        masks_bbox = masks_bbox.numpy()*255
        masks_bbox = cv2.cvtColor(masks_bbox, cv2.COLOR_BGR2RGB)
        print("masks bbox shape: ",masks_bbox.shape)
        fontScale = min(masks_bbox.shape[1], masks_bbox.shape[0]) * 2e-3
        masks_bbox = cv2.putText(masks_bbox, str(iou), (250,250), cv2.FONT_HERSHEY_SIMPLEX, int(fontScale), (0, 0, 255), 5, cv2.LINE_AA)
        print("mask bbox: ", masks_bbox.shape)
        cv2.imwrite(mask_bbox_path, masks_bbox)
        #torchvision.utils.save_image(masks_bbox, mask_bbox_path, normalize=True, nrow=1, range=(0, 1))
        if i%1000 ==0:
            print(i)
            #print("bbox: ",train_dataset.count)
    #print("final overlap>=0.9 count: ",count)
    #print("final per pcb: ")
    #print(holder)
    print("count: ",train_dataset.count)

    
    
    

    

