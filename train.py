#### Current project files
from dataParser import PCBParser
from datasetSegmentation import PCBDataset
# from losses import SegmentationLosses, PatchTverskyLoss, BinaryTverskyLoss, PatchBinaryTverskyLoss
from losses import TverskyLoss, PatchTverskyLoss
import visualizer
import Config.configuration as cfg
import Config.parameters as params
#from model import LinkNet
#from linknet import LinkNet
from Model.deeplabv3 import DeepLabV3
import metrices
from augmentation import Augmentation
from profiler import Profiler
from validation import validation



#### Third party libraries
from tensorboardX import SummaryWriter

#### torch library
import torch  
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchgeometry as tgm

#### python library
from collections import OrderedDict
import os
import numpy as np
from datetime import datetime
import time
import sys


# -------------------------------------------- Global Variable Declaration
device = torch.device('cuda:'+str(params.gpu))
# --------------------------------------------

torch.set_flush_denormal(True)

# Training
def train(run_id, use_cuda, epoch, train_dataloader, model, optimizer, segmentationLoss, writer):
    print('Training Epoch: %d' % epoch)

    global device   #will be needed while training on 1 GPU
    model.train()
    total_losses = []
    dice_losses = []
    bce_losses = []
    iou_epoch = []
    dice_coeff_epoch = []
    start_time = time.time()
    #save_epoch = (epoch % params.visualize_epoch_freq == 0)

    for param_group in optimizer.param_groups:
        print('Learning rate: ',param_group['lr'])
        param_group['lr'] = params.learning_rate
    # for param_group in optimizer.param_groups:
    #     print('Learning rate: ',param_group['lr'])
    #     param_group['lr'] = params.learning_rate

    #profiler = Profiler(summarize_every=10, disabled=False)
    batch_start_time = start_time
    for i, (inputs, masks, masks_bbox, bbox_label) in enumerate(train_dataloader):

        #profiler.tick("Blocking, waiting for batch (threaded)")
        if use_cuda and not params.device_cpu:
            #print('loading intput into GPU')
            ########### While working with Single GPU
            """ inputs, targets = inputs.to(device), targets.to(device)
            classification_loss.to(device) """
            ################################
            ##############While working with Multiple GPU
            inputs, masks = inputs.to(f'cuda:{model.device_ids[0]}', non_blocking=True), masks.to(f'cuda:{model.device_ids[0]}', non_blocking=True)
            masks_bbox = masks_bbox.to(f'cuda:{model.device_ids[0]}', non_blocking=True)
            bbox_label = bbox_label.to(f'cuda:{model.device_ids[0]}', non_blocking=True)
            #profiler.tick("Data to %s" % device)
            segmentationLoss.to(f'cuda:{model.device_ids[0]}')
            ################

        ##train
        optimizer.zero_grad()
        outputs = model(inputs)
        # Calculate loss with bbox mask
        # loss, patch_tv_loss, bce_loss = segmentationLoss(outputs, masks_bbox)
        loss,_,_ = segmentationLoss(outputs, masks_bbox)
        loss.backward()
        optimizer.step()

        ##store statistics
        total_losses.append(loss.item())
        # dice_losses.append(patch_tv_loss.item())
        # bce_losses.append(bce_loss.item())

        # Calculate Metrices with actual mask
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

        if i%params.print_stats == 0:
            time_taken = time.time() - batch_start_time
            m, s = divmod(time_taken, 60)
            h, m = divmod(m, 60)
            # print(
            #     "[Epoch %d/%d] [Batch %d/%d] [loss: %f] [patch_tv_loss:%f] [bce_loss:%f] [iou: %f] [dice_coeff:%f]"
            #     % (epoch, params.num_epoch, i, len(train_dataloader), np.mean(total_losses), np.mean(dice_losses), np.mean(bce_losses), np.mean(iou_epoch), np.mean(dice_coeff_epoch))
            # )
            print(
                "[Epoch %d/%d] [Batch %d/%d] [loss: %f]  [iou: %f] [dice_coeff:%f]"
                % (epoch, params.num_epoch, i, len(train_dataloader), np.mean(total_losses), np.mean(iou_epoch), np.mean(dice_coeff_epoch))
            )
            if i==0:
                print(f'time taken for 1st batch-h:{h} m:{m} s:{s}')
            else:
                print(f'time taken for {params.print_stats} batch-h:{h} m:{m} s:{s}')
            batch_start_time = time.time()
            
            current_iteration = epoch*len(train_dataloader) + i
            writer.add_scalar('Training Loss per 500 iteration', np.mean(total_losses), current_iteration)
            # writer.add_scalar('Training Patch TV Loss per 500 iteration', np.mean(dice_losses), current_iteration)
            # writer.add_scalar('Training BCE Loss per 500 iteration', np.mean(bce_losses), current_iteration)
            writer.add_scalar('IOU per 500 iteration', np.mean(iou_epoch), current_iteration)
            writer.add_scalar('Dice Coeff per 500 iteration', np.mean(dice_coeff_epoch), current_iteration)
            #profiler.tick("Logging")

        if i%params.visualize_mask==0:
            if len(masks.shape)==3:
                masks = torch.unsqueeze(masks, dim=1).float()
            if len(pred_labels.shape)==3:
                pred_labels = torch.unsqueeze(pred_labels, dim=1)
            visuals = OrderedDict([('input', inputs[0:8, :, :, :]),
                                ('mask', masks[0:8, :, :, :]),
                                ('output', pred_labels[0:8, :, :, :])])

            visualizer.write_img(visuals, run_id, epoch, i)
            #profiler.tick("saving images")


    save_dir = os.path.join(cfg.saved_models_dir, run_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if epoch % params.checkpoint == 0:
        save_file_path = os.path.join(save_dir, 'model_{}.pth'.format(epoch))
        states = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)
        #profiler.tick("saving models")
    time_taken = time.time() - start_time
    m, s = divmod(time_taken, 60)
    h, m = divmod(m, 60)
    # print(f'Training Epoch {epoch}::: Total Loss:{np.mean(total_losses)} Dice/TV Loss:{np.mean(dice_losses)} BCE/CE Loss:{np.mean(bce_losses)} Mean IOU:{np.mean(iou_epoch)} Mean Dice Coeff:{np.mean(dice_coeff_epoch)} time taken-h:{h} m:{m} s:{s}')
    print(f'Training Epoch {epoch}::: Total Loss:{np.mean(total_losses)} Mean IOU:{np.mean(iou_epoch)} Mean Dice Coeff:{np.mean(dice_coeff_epoch)} time taken-h:{h} m:{m} s:{s}')
    sys.stdout.flush()
    writer.add_scalar('Training Total Loss', np.mean(total_losses), epoch)
    # writer.add_scalar('Training Patch TV Loss', np.mean(dice_losses), epoch)
    # writer.add_scalar('Training BCE Loss', np.mean(bce_losses), epoch)
    writer.add_scalar('Training Mean IOU', np.mean(iou_epoch), epoch)
    writer.add_scalar('Training Mean D_Coeff', np.mean(dice_coeff_epoch), epoch)
    #profiler.tick("final logging")
    return model


def train_task(run_id, use_cuda):
    global device
    device_cpu = params.device_cpu
    

    if params.resume:
        index = cfg.resume_model.rindex('/')
        resume_id = str(cfg.resume_model[index-4:index])
        run_id = run_id + '_resume_'+ resume_id

    print('run id: ',run_id)

    writer = SummaryWriter(os.path.join(cfg.logs_dir, str(run_id)))
    model = DeepLabV3(num_classes=2)

    if use_cuda and not device_cpu:
        print('loading model into GPU')
        #######To load in Single GPU
        #model.to(device)
        #######

        #####To load in Multiple GPU. In FICS server we have 4 GPU. That's why 4 device ids here.
        model = nn.DataParallel(model, device_ids = params.device_ids)
        model.to(f'cuda:{model.device_ids[0]}', non_blocking=True)
        ######
    else:
        print('model will be trained on CPU')

    augImg = Augmentation()
    data_parser = PCBParser(cfg.data_path, run_id)
    train_dataset = PCBDataset(data_parser.img_files, transform=augImg)
    print('length train dataset: ',len(train_dataset))

    val_parser = PCBParser(cfg.data_path, run_id, val=True)
    val_dataset = PCBDataset(val_parser.img_files, val=True)
    print('length val dataset: ',len(val_dataset))
    
    segmentationLoss = PatchTverskyLoss(cuda=True, device_ids=model.device_ids)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    
    if params.resume:
        print('##########loading pre-trained model###########')
        saved_states = torch.load(cfg.resume_model)
        model.load_state_dict(saved_states['state_dict'])
        optimizer.load_state_dict(saved_states['optimizer'])
        print('##########Done###########')


    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)

    for epoch in range(params.num_epoch):
        train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, num_workers=4, pin_memory=True, drop_last=True, shuffle=True)
        print("train dataloader: ",len(train_dataloader))
        #exit()
        model = train(run_id, use_cuda, epoch, train_dataloader, model, optimizer, segmentationLoss, writer)
        
        val_dataloader = DataLoader(val_dataset, batch_size=params.batch_size, num_workers=4, pin_memory=True, drop_last=True, shuffle=True)
        print("validation dataloader: ",len(val_dataloader))
        #exit()
        mean_dice_coeff, mean_iou, mean_loss  = validation(run_id, use_cuda, epoch, val_dataloader, model, segmentationLoss, writer)
        #scheduler.step(mean_loss)

if __name__== "__main__":

    run_started = datetime.today().strftime('%m-%d-%y_%H%M')
    #print('run id: ',run_started)
    use_cuda = torch.cuda.is_available()
    print('use cuda: ',use_cuda)
    train_task(run_started, use_cuda)
