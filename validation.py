# PyTorch
import torch
import time


#### Current project files
import visualizer
import Config.parameters as params
import metrices


#### python library
from collections import OrderedDict
import os
import numpy as np
from datetime import datetime
import time
import sys


def validation(run_id, use_cuda, epoch, val_dataloader, model, segmentationLoss, writer):
    print('Validation Epoch: %d' % epoch)
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
        for i, (inputs, masks) in enumerate(val_dataloader):

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
            loss, _, _ = segmentationLoss(outputs, masks)
            # loss = segmentationLoss(outputs, masks)

            ##store statistics
            total_losses.append(loss.item())
            # dice_losses.append(patch_tv_loss.item())
            # bce_losses.append(bce_loss.item())

            # outputs --> B x #num_class x H x W more than one class
            if outputs.shape[1]>1:
                predictions = torch.nn.functional.softmax(outputs, dim=1)
                pred_labels = torch.argmax(predictions, dim=1) 
                pred_labels = pred_labels.float()   #B x H x W
                # pred_labels = pred_labels.type(torch.LongTensor)
                #print(type(pred_labels), type(masks))

            # outputs --> B x 1 x H x W  for 2 class in binary class fashion
            else:
                pred_labels = (torch.sigmoid(outputs)>params.train_conf_th).float()

            iou = metrices.iou_score(pred_labels, masks)
            d_coeff = metrices.dice_coefficient(pred_labels, masks)
            iou_epoch.append(iou)
            dice_coeff_epoch.append(d_coeff)

            if i%params.print_stats_val == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [loss: %f] [iou: %f] [dice_coeff:%f]"
                    % (epoch, params.num_epoch, i, len(val_dataloader), np.mean(total_losses), np.mean(iou_epoch), np.mean(dice_coeff_epoch))
                )

            if i%params.visualize_mask_val==0:
                if len(masks.shape)==3:
                    masks = torch.unsqueeze(masks, dim=1).float()
                if len(pred_labels.shape)==3:
                    pred_labels = torch.unsqueeze(pred_labels, dim=1)

                visuals = OrderedDict([('input', inputs[0:8, :, :, :]),
                                    ('mask', masks[0:8, :, :, :]),
                                    ('output', pred_labels[0:8, :, :, :])])
                visualizer.write_img(visuals, run_id, epoch, i, val=True)

    time_taken = time.time() - start_time
    m, s = divmod(time_taken, 60)
    h, m = divmod(m, 60)
    # print(f'Validation Epoch {epoch}::: Total Loss:{np.mean(total_losses)} Dice/TV Loss:{np.mean(dice_losses)} BCE/CE Loss:{np.mean(bce_losses)} Mean IOU:{np.mean(iou_epoch)} Mean Dice Coeff:{np.mean(dice_coeff_epoch)} time taken-h:{h} m:{m} s:{s}')
    print(f'Validation Epoch {epoch}::: Total Loss:{np.mean(total_losses)} Mean IOU:{np.mean(iou_epoch)} Mean Dice Coeff:{np.mean(dice_coeff_epoch)} time taken-h:{h} m:{m} s:{s}')
    sys.stdout.flush()
    writer.add_scalar('Validation Total Loss', np.mean(total_losses), epoch)
    # writer.add_scalar('Validation Patch TV Loss', np.mean(dice_losses), epoch)
    # writer.add_scalar('Validation BCE Loss', np.mean(bce_losses), epoch)
    writer.add_scalar('Validation Mean IOU', np.mean(iou_epoch), epoch)
    writer.add_scalar('Validation Mean D_Coeff', np.mean(dice_coeff_epoch), epoch)
    return np.mean(dice_coeff_epoch), np.mean(iou_epoch), np.mean(total_losses)