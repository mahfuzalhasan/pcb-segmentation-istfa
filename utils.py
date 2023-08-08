
import numpy as np
import cv2
import os
import copy
import torch

def one_hot(labels,
            num_classes: int,
            device = None,
            dtype = None,
            eps = 1e-6):
    
    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                        .format(type(labels)))
    if not len(labels.shape) == 3:
        raise ValueError("Invalid depth shape, we expect BxHxW. Got: {}"
                         .format(labels.shape))
    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}" .format(
                labels.dtype))
    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))
    batch_size, height, width = labels.shape
    one_hot = torch.zeros(batch_size, num_classes, height, width,
                          device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


def bm1(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    mask1_area = np.count_nonzero( mask1 )
    mask2_area = np.count_nonzero( mask2 )
    intersection = np.count_nonzero( np.logical_and( mask1, mask2 ) )
    iou = intersection/(mask1_area+mask2_area-intersection+1e-6)
    return iou

def convert_to_bbox_mask(mask):
    #print(mask.shape)
    """ mask_copied = copy.deepcopy(mask[:,:,0])
    mask_2 = copy.deepcopy(mask)
    #cv2.imwrite(os.path.join('data','mask_1ch.jpg'), mask)
    contours = cv2.findContours(mask_copied, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(len(contours))
    contours = contours[0] if len(contours) == 2 else contours[1]
    print(len(contours))
    #exit() """

    #mask_copied = copy.deepcopy(mask[:, :, 0])
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    for cntr in contours:
        # x,y,w,h = cv2.boundingRect(cntr)
        # if w < 10 or h < 10:
        #     continue

        # cv2.rectangle(mask, (x, y), (x+w, y+h), (0, 0, 255), 2)
        # print("x,y,w,h:",x,y,w,h)

        rect = cv2.minAreaRect(cntr)
        box = np.int0(cv2.boxPoints(rect))
        #cv2.drawContours(mask_2, [box], 0, (255, 0, 0), 2)
        cv2.fillPoly(mask, pts = [box], color =(255, 255, 255))

    #cv2.imwrite(os.path.join('data','mask_boxed.jpg'), mask)
    #cv2.imwrite(os.path.join('data','filled_mask_1ch.jpg'), mask_copied)
    return mask


if __name__=='__main__':
    mask = cv2.imread('./data/mask.jpg')
    convert_to_bbox_mask(mask)
