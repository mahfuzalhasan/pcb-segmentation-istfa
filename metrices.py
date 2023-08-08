import torch
import torchvision
from torchvision.utils import draw_segmentation_masks

def iou_accuracy(predicted, targets, threshold=0.7, epsilon=1e-6):
    # Target --> BxCxHxW    C = #num_class
    # pred --> BxCXHxW   C = #num_class 
    if len(predicted.shape)==4:
        axis = (1, 2, 3)
    # Target --> BxHxW
    # pred --> BxHxW
    if len(predicted.shape)==3:
        axis = (1, 2)
    intersection = (predicted.long() & targets.long()).float().sum(axis)
    union = (predicted.long() | targets.long()).float().sum(axis)
    return torch.mean(intersection/(union + epsilon)).item()

def iou_score(Y_predicted, Y_true):
    Y_true = Y_true.float()
    Y_true = Y_true.view(-1)
    Y_predicted = Y_predicted.view(-1)
    Y_predicted = Y_predicted.float()
    intersection = torch.logical_and(Y_predicted, Y_true).sum()
    union = torch.logical_or(Y_predicted, Y_true).sum()
    iou = (intersection.float() + 1) / (union.float() + 1)
    return iou.item()

def dice_coefficient(Y_predicted, Y_true, smoothness=1.0):
    Y_true = Y_true.float()
    Y_true = Y_true.view(-1)
    Y_predicted = Y_predicted.view(-1)
    Y_predicted = Y_predicted.float()
    intersection = torch.sum(Y_true * Y_predicted)
    dc = (2.0 * intersection + smoothness) / (
        torch.sum(Y_true) + torch.sum(Y_predicted) + smoothness
    )
    return dc.item()



# def dice_coefficient(predicted, targets, threshold=0.7, epsilon=1):
#     # Target --> BxCxHxW    
#     # pred --> BxCXHxW   
#     if len(predicted.shape)==4:
#         axis = (1, 2, 3)
#     # Target --> BxHxW
#     # pred --> BxHxW
#     if len(predicted.shape)==3:
#         axis = (1, 2)   
#     intersection = torch.sum(predicted * targets, dim = axis)
#     cardinal = torch.sum(predicted + targets, dim = axis)
#     return torch.mean((2 * intersection + epsilon) / (cardinal + epsilon)).item()

def overlay_mask(inputs, normalized_masks, num_classes):
    class_dim = 1
    inputs = (inputs * 255.0).type(torch.uint8)
    all_classes_masks = normalized_masks.argmax(class_dim) == torch.arange(num_classes)[:, None, None, None]
    print(f"shape = {all_classes_masks.shape}, dtype = {all_classes_masks.dtype}")

    # The first dimension is the classes now, so we need to swap it
    all_classes_masks = all_classes_masks.swapaxes(0, 1)

    dogs_with_masks = [
        draw_segmentation_masks(img, masks=mask, alpha=.6)
        for img, mask in zip(inputs, all_classes_masks)
    ]

    dogs_with_masks = torch.stack(dogs_with_masks)
    # print(dogs_with_masks.size())
    # pred_labels = torch.argmax(normalized_masks, dim=1) 
    # pred_labels = pred_labels.float()
    # pred_labels = torch.unsqueeze(pred_labels, dim=1)

    #torchvision.utils.save_image(pred_labels,'./Results/preds.jpg',  normalize=True, nrow=4, range=(0, 1))
    #torchvision.utils.save_image(inputs.type(torch.float32),'./Results/inputs.jpg',  normalize=False, nrow=4)
    torchvision.utils.save_image(dogs_with_masks.type(torch.float32),'./Results/overlay.jpg',  normalize=False, nrow=4)

if __name__=='__main__':
    inputs = torch.randn(8, 3, 512, 512)
    outputs = torch.randn(8, 2, 512, 512)
    inputs = (inputs * 255.0).type(torch.uint8)

    pred_masks = torch.nn.functional.softmax(outputs, dim=1)
    overlay_mask(inputs, pred_masks, 2)
