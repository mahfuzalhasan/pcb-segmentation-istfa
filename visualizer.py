from tensorboardX import SummaryWriter
import os
import torchvision
import math
import numpy as np
import cv2

import torch
from torchvision.utils import draw_segmentation_masks

import Config.configuration as cfg
import Config.parameters as params




""" def overlay_mask(inputs, normalized_masks, num_classes):
    class_dim = 1
    device='cpu'
    inputs = (inputs * 255.0).type(torch.uint8)
    #gpu_id = normalized_masks.get_device()
    normalized_masks = normalized_masks.to(device)
    inputs = inputs.to(device)
    #device = 'cuda:'+str(gpu_id)
    
    class_broadcast = torch.arange(num_classes)[:, None, None, None]
    #class_broadcast = class_broadcast.to(device)
    
    all_classes_masks = normalized_masks.argmax(class_dim) == class_broadcast
    # print(f"shape = {all_classes_masks.shape}, dtype = {all_classes_masks.dtype}")

    # The first dimension is the classes now, so we need to swap it
    all_classes_masks = all_classes_masks.swapaxes(0, 1)

    color = (240, 0, 0)

    inputs_with_masks = [
        draw_segmentation_masks(img, masks=mask, alpha=.6, colors=color)
        for img, mask in zip(inputs, all_classes_masks)
    ]
    inputs_with_masks = torch.stack(inputs_with_masks)

    return inputs_with_masks.type(torch.float32) """
    
def overlay_mask(inputs, outputs):
    inputs = torch.permute(inputs, (0, 2, 3, 1))
    outputs = torch.permute(outputs, (0, 2, 3, 1))

    inputs = inputs.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    masks = []
    for img, mask in zip(inputs, outputs):
        img = (img*255).astype(np.uint8)
        mask = (mask*255).astype(np.uint8)
        mask = np.stack((mask,)*3, axis=2).squeeze()
        dst = cv2.addWeighted(img, 1, mask, 0.5, 0)
        masks.append(torch.from_numpy(dst)/255.0)
    masks = torch.stack(masks)
    masks = torch.permute(masks, (0, 3, 1, 2))
    return masks





def write_img(visuals, run_id, ep, iteration, val=False, test=False ):
    if not val:
        if not test:
            path = os.path.join(cfg.output_dir, run_id, 'train', str(ep))
        else:
            folder_id = int(math.floor(iteration/100))
            path = os.path.join(cfg.output_dir, run_id, 'test', str(folder_id))

    else:
        path = os.path.join(cfg.output_dir, run_id, 'val', str(ep))
        
    if not os.path.exists(path):
        os.makedirs(path)

    input_img = '%s/%05d_input.jpg' % (path, iteration)
    mask = '%s/%05d_mask.jpg' % (path, iteration)
    output = '%s/%05d_output.jpg' % (path, iteration)
    overlay = '%s/%05d_overlay.jpg' % (path, iteration)
    
    

    n_row = 1 if test else 4
    
    torchvision.utils.save_image(visuals['input'], input_img, normalize=True, nrow=n_row, range=(0, 1))
    torchvision.utils.save_image(visuals['mask'], mask, normalize=True, nrow=n_row, range=(0, 1))
    torchvision.utils.save_image(visuals['output'], output, normalize=True, nrow=n_row, range=(0, 1))
    
    if test:
        input_with_masks = overlay_mask(visuals['input'], visuals['output'])
        torchvision.utils.save_image(input_with_masks, overlay, normalize=True, nrow=n_row, range=(0, 1))
    



