import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchgeometry as tgm

import numpy as np
import time
from collections import OrderedDict
import sys


from dataParser import PCBParser
from datasetSegmentation import PCBDataset
from losses import BinaryTverskyLoss, TverskyLoss, PatchTverskyLoss
import visualizer
import Config.configuration as cfg
import Config.parameters as params

#from model import LinkNet
#from model import LinkNet
# from linknet import LinkNet
from Model.deeplabv3 import DeepLabV3

import metrices
from augmentation import Augmentation
from profiler import Profiler


def test(run_id, use_cuda, epoch, test_dataloader, model, segmentationLoss):
    print('Testing')
    global device   #while using 1 GPU
    model.eval()
    total_losses = []
    dice_losses = []
    bce_losses = []
    iou_epoch = []
    dice_coeff_epoch = []
    start_time = time.time()
    #save_epoch = (epoch % params.visualize_epoch_freq == 0)
    with torch.no_grad():
        for i, (inputs, masks) in enumerate(test_dataloader):
            # if i%100==0:
            #     print(f'step:{i}')
            if use_cuda and not params.device_cpu:
                #print('loading intput into GPU')
                ########### While working with Single GPU
                """ inputs, targets = inputs.to(device), targets.to(device)
                classification_loss.to(device) """
                ################################
                ##############While working with Multiple GPU
                inputs, masks = inputs.to(f'cuda:{model.device_ids[0]}', non_blocking=True), masks.to(f'cuda:{model.device_ids[0]}', non_blocking=True)
                segmentationLoss.to(f'cuda:{model.device_ids[0]}')
                ################

            outputs = model(inputs)
            #loss, dice_loss, bce_loss = segmentationLoss(outputs, masks)
            loss, _, _ = segmentationLoss(outputs, masks)

            ##store statistics
            total_losses.append(loss.item())
            #dice_losses.append(dice_loss.item())
            #bce_losses.append(bce_loss.item())

            # outputs --> B x num_class x H x W 
            if outputs.shape[1]>1:
                predictions = torch.nn.functional.softmax(outputs, dim=1)
                pred_labels = torch.argmax(predictions, dim=1) 
                pred_labels = pred_labels.float()   #B x H x W
                # pred_labels = pred_labels.type(torch.LongTensor)
                #print(type(pred_labels), type(masks))

            # outputs --> B x 1 x H x W  for 2 class in binary fashion
            else:
                pred_labels = (torch.sigmoid(outputs)>params.test_conf_th).float()


            iou = metrices.iou_score(pred_labels, masks)
            d_coeff = metrices.dice_coefficient(pred_labels, masks)
            iou_epoch.append(iou)
            dice_coeff_epoch.append(d_coeff)

            """ if i%params.print_stats_val == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [loss: %f] [dice_loss:%f] [bce_loss:%f] [iou: %f] [dice_coeff:%f]"
                    % (epoch, params.num_epoch, i, len(test_dataloader), np.mean(total_losses), np.mean(dice_losses), np.mean(bce_losses), np.mean(iou_epoch), np.mean(dice_coeff_epoch))
                ) """

            if i%params.visualize_mask_test==0:
                if len(masks.shape)==3:
                    masks = torch.unsqueeze(masks, dim=1).float()
                if len(pred_labels.shape)==3:
                    pred_labels = torch.unsqueeze(pred_labels, dim=1)

                visuals = OrderedDict([('input', inputs[0:8, :, :, :]),
                                    ('mask', masks[0:8, :, :, :]),
                                    ('output', pred_labels[0:8, :, :, :])])
                visualizer.write_img(visuals, run_id, epoch, i, test=True)

    time_taken = time.time() - start_time
    m, s = divmod(time_taken, 60)
    h, m = divmod(m, 60)
    print(f'Test ::: Total Loss:{np.mean(total_losses)} Mean IOU:{np.mean(iou_epoch)} Mean Dice Coeff:{np.mean(dice_coeff_epoch)} time taken-h:{h} m:{m} s:{s}')
    sys.stdout.flush()
    return np.mean(dice_coeff_epoch), np.mean(iou_epoch), np.mean(total_losses)


def test_performance(saved_model, use_cuda):
    global device
    device_cpu = params.device_cpu
    
    index_1 = saved_model.rindex('/')
    string_1 = saved_model[:index_1]
    index_2 = string_1.rindex('/')
    run_id = string_1[index_2+1:]
    # run_id = saved_model[index-13:index]
    # run_id = saved_model
    print("saved model: ",saved_model)
    print("run id: ",run_id)
    # exit()
    
    model = DeepLabV3(num_classes=2)
    if use_cuda and not device_cpu:
        print('loading model into GPU')
        #######To load in Single GPU
        #model.to(device)
        #######

        #####To load in Multiple GPU. In FICS server we have 4 GPU. That's why 4 device ids here.
        model = nn.DataParallel(model, device_ids = [2, 3])
        model.to(f'cuda:{model.device_ids[0]}', non_blocking=True)
        ######
    else:
        print('model will be trained on CPU')
    
    #segmentationLoss = SegmentationLosses(cuda=use_cuda, device_ids=model.device_ids)
    segmentationLoss = PatchTverskyLoss(cuda=True, device_ids=model.device_ids)

    saved_states = torch.load(saved_model)
    model.load_state_dict(saved_states['state_dict'])

    test_parser = PCBParser(cfg.data_path, run_id, test=True)
    test_dataset = PCBDataset(test_parser.img_files, val=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=4, pin_memory=True, shuffle=False)
    print("test dataset: ", len(test_dataset))
    #exit()

    epoch = 0
    mean_dice, mean_iou, mean_loss = test(run_id, use_cuda, epoch, test_dataloader, model, segmentationLoss)
    print(f'mean_dice:{mean_dice} mean_iou:{mean_iou} mean_loss:{mean_loss}')


if __name__=='__main__':
    use_cuda = torch.cuda.is_available()
    print('use cuda: ',use_cuda)
    test_performance(cfg.test_model_path, use_cuda)