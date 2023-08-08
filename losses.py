import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import resnet

import torchgeometry as tgm

from utils import one_hot
from einops import rearrange


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.loss = tgm.losses.TverskyLoss(alpha=0.3, beta=0.7)

    def forward(self, predicted, targets):
        return self.loss(predicted, targets)


class TverskyLoss(nn.Module):

    def __init__(self, alpha, beta):
        super(TverskyLoss, self).__init__()
        self.alpha: float = alpha
        self.beta: float = beta
        self.eps: float = 1e-6

    def forward(
            self,
            input,
            target):
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1)

        # create the labels one hot tensor
        target_one_hot = one_hot(target, num_classes=input.shape[1],
                                 device=input.device, dtype=input.dtype)

        # compute the actual dice score
        dims = (2, 3)
        intersection = torch.sum(input_soft * target_one_hot, dims)
        fps = torch.sum(input_soft * (1. - target_one_hot), dims)
        fns = torch.sum((1. - input_soft) * target_one_hot, dims)
        #print('dim 2 3',intersection.shape, fps.shape, fns.shape)

        numerator = intersection
        denominator = intersection + self.alpha * fps + self.beta * fns
        tversky_loss = numerator / (denominator + self.eps)
        #print('tv loss dim 2 3: ',tversky_loss, tversky_loss.size())

        #print('dim 2 3 final: ', torch.mean(1-tversky_loss))

        dims = (1, 2, 3)
        intersection = torch.sum(input_soft * target_one_hot, dims)
        fps = torch.sum(input_soft * (1. - target_one_hot), dims)
        fns = torch.sum((1. - input_soft) * target_one_hot, dims)
        #print('dim 123', intersection.shape, fps.shape, fns.shape)

        numerator = intersection
        denominator = intersection + self.alpha * fps + self.beta * fns
        tversky_loss = numerator / (denominator + self.eps)
        #print('tv loss dim 123: ',tversky_loss, tversky_loss.size())
        #print('dim 1 2 3 final: ', torch.mean(1-tversky_loss))
        return torch.mean(1. - tversky_loss)



class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class BinaryTverskyLoss(nn.Module):
    def __init__(self, delta = 0.7, gamma = 3, size_average=True):
        super(BinaryTverskyLoss, self).__init__()
        self.size_average = size_average
        self.delta = delta
        self.gamma = gamma

    # preds --> B x 1 x H x W
    # targets --> B x 1 x H x W where targets[:,:,:,:] = 0/1
    def forward(self, preds, targets):

        N = preds.size(0)   
        P = F.sigmoid(preds)
        
        P = P.permute(0, 2, 3, 1).contiguous().view(-1, 1)
        class_mask = targets.permute(0, 2, 3, 1).contiguous().view(-1, 1)
        
        smooth = torch.zeros(1, dtype=torch.float32).fill_(0.00001)
        smooth = smooth.to(P.device)


        ones = torch.ones(P.shape).to(P.device)
        P_ = ones - P
        class_mask_ = ones - class_mask

        TP = P * class_mask
        FP = P * class_mask_
        FN = P_ * class_mask
        
        # self.beta = 1 - self.delta
        num = torch.sum(TP, dim=(0)).float()
        den = num + self.delta * torch.sum(FP, dim=(0)).float() + (1-self.delta) * torch.sum(FN, dim=(0)).float()

        TI = num / (den + smooth)

        if self.size_average:       
            ones = torch.ones(TI.shape).to(TI.device)
            loss = ones - TI.mean()
        else:
            TI = TI.sum()
            # ones = torch.ones(TI.shape).to(TI.device)
            loss = 1. - TI

        # loss = 1. - TI
        # asym_focal_tl = torch.where(torch.eq(class_mask, 1), loss.pow(1-self.gamma), loss)
        
        return loss


class PatchBinaryTverskyLoss(nn.Module):
    def __init__(self, cuda=False, device_ids = None):
        super(PatchBinaryTverskyLoss, self).__init__()
        # self.ignore_index = ignore_index
        # self.weight = weight
        # self.size_average = size_average
        self.cuda = cuda
        self.device_ids = device_ids 
        self.binary_tversky_loss = BinaryTverskyLoss(delta=0.7, gamma=3, size_average=False)
        if cuda:
            self.binary_tversky_loss.to(self.device_ids[0])
    def forward(self, logit, target):
        n, c, h, w = logit.size()
        smooth = 1
        probas = logit
        true_1_hot = target
        
        logit_5x5 = probas.unfold(2,5,5).unfold(3,5,5).contiguous()
        logit_5x5 = logit_5x5.view(n, -1, 5, 5)
        logit_7x7 = probas.unfold(2,7,7).unfold(3,7,7).contiguous()
        logit_7x7 = logit_7x7.view(n, -1, 7, 7)

        target_5x5 = true_1_hot.unfold(2,5,5).unfold(3,5,5).contiguous()
        target_5x5 = target_5x5.view(n, -1, 5, 5)
        target_7x7 = true_1_hot.unfold(2,7,7).unfold(3,7,7).contiguous()
        target_7x7 = target_7x7.view(n, -1, 7, 7)

        loss_5_5 = self.binary_tversky_loss(logit_5x5, target_5x5)
        loss_7_7 = self.binary_tversky_loss(logit_7x7, target_7x7)

        criterion = nn.BCEWithLogitsLoss()

        if self.cuda:
            criterion = criterion.to(f'cuda:{self.device_ids[0]}')

        bce_loss = criterion(logit, target)

        return loss_5_5 + loss_7_7 + 2*bce_loss, loss_5_5+loss_7_7, bce_loss



class PatchTverskyLoss(nn.Module):
    def __init__(self, cuda=False, device_ids = None):
        super(PatchTverskyLoss, self).__init__()
        # self.ignore_index = ignore_index
        # self.weight = weight
        # self.size_average = size_average
        self.cuda = cuda
        self.device_ids = device_ids 
        self.patch_tversky_loss = TverskyLoss(alpha=0.3, beta=0.7)
        self.ce_image_loss = nn.CrossEntropyLoss()

        if self.cuda:
            self.patch_tversky_loss = self.patch_tversky_loss.to(f'cuda:{self.device_ids[0]}')
            self.ce_image_loss = self.ce_image_loss.to(f'cuda:{self.device_ids[0]}')
            


    # logit == unnormalized(no activation like softmax applied) predictions
    # logit --> B, C, H, W  where C = #class
    # target --> B, H, W where  0<=target[i]<=C-1 
    def forward(self, logit, target):
        n, c, _, _ = logit.size()
        true_1_hot = target
        
        logit_5x5 = logit.unfold(2, 5, 5).unfold(3, 5, 5).contiguous()
        #print('logit 5x5: ',logit_5x5.size())
        logit_5x5 = logit_5x5.view(n, c, -1, 5, 5)
        logit_5x5 = torch.permute(logit_5x5, (0, 2, 1, 3, 4)).flatten(0, 1)
        #print('logit 5x5 flatten: ',logit_5x5.size())
        logit_7x7 = logit.unfold(2, 7, 7).unfold(3, 7, 7).contiguous()
        #print('logit 7x7: ',logit_7x7.size())
        logit_7x7 = logit_7x7.view(n, c, -1, 7, 7)
        logit_7x7 = torch.permute(logit_7x7, (0, 2, 1, 3, 4)).flatten(0, 1)
        #print('logit 7x7 flatten: ',logit_7x7.size())

        target_5x5 = true_1_hot.unfold(1, 5, 5).unfold(2, 5, 5).contiguous()
        #print('target_5x5: ',target_5x5.size())
        target_5x5 = target_5x5.view(n, -1, 5, 5).flatten(0, 1)
        #print('target_5x5 flatten: ',target_5x5.size())
        target_7x7 = true_1_hot.unfold(1, 7, 7).unfold(2, 7, 7).contiguous()
        #print('target_7x7: ',target_7x7.size())
        target_7x7 = target_7x7.view(n, -1, 7, 7).flatten(0, 1)
        #print('target_7x7 flatten: ',target_7x7.size())

        # Patch_Loss
        loss_5x5 = self.patch_tversky_loss(logit_5x5, target_5x5)
        loss_7x7 = self.patch_tversky_loss(logit_7x7, target_7x7)
        #print('losses: ',loss_5x5, loss_7x7)
        # CE on whole image
        ce_loss = self.ce_image_loss(logit, target)

        return loss_5x5+loss_7x7+2*ce_loss, loss_5x5+loss_7x7, ce_loss


class SegmentationLosses(nn.Module):
    def __init__(self, cuda=True, device_ids = None):
        super(SegmentationLosses, self).__init__()
        # self.ignore_index = ignore_index
        # self.weight = weight
        # self.size_average = size_average
        self.cuda = cuda
        self.device_ids = device_ids    

    def forward(self, logit, target):
        n, c, h, w = logit.size()
        smooth = 1
        probas = torch.sigmoid(logit)
        true_1_hot = target
        
        logit_5x5 = probas.unfold(2,5,5).unfold(3,5,5).contiguous()
        ##print('logit 5x5: ',logit_5x5.size())
        logit_5x5 = logit_5x5.view(n, -1, 5, 5)
        ##print('logit 5x5 flatten: ',logit_5x5.size())
        logit_7x7 = probas.unfold(2,7,7).unfold(3,7,7).contiguous()
        ##print('logit 7x7: ',logit_7x7.size())
        logit_7x7 = logit_7x7.view(n, -1, 7, 7)
        ##print('logit 7x7 flatten: ',logit_7x7.size())

        target_5x5 = true_1_hot.unfold(2,5,5).unfold(3,5,5).contiguous()
        target_5x5 = target_5x5.view(n, -1, 5, 5)
        target_7x7 = true_1_hot.unfold(2,7,7).unfold(3,7,7).contiguous()
        target_7x7 = target_7x7.view(n, -1, 7, 7)

        # pdb.set_trace()

        def dice_loss(current_logit, current_target, smooth=smooth):
            dims = (0,) + tuple(range(2, current_target.ndimension()))
            #dims = (1,2,3)
            ##print('dims: ',dims)
            intersection = torch.sum(current_logit * current_target, dims)
            ##print('intersection: ',intersection.size())
            cardinality = torch.sum(current_logit + current_target, dims)
            ##print('cardinality: ',cardinality.size())
            dice_Loss = ((2. * intersection + smooth) / (cardinality + smooth)).mean()
            return 1-dice_Loss


        # loss_3_3 = dice_loss(logit_3x3, target_3x3)
        # pdb.set_trace()
        loss_5_5 = dice_loss(logit_5x5, target_5x5)
        #print("############## for patch 7x7##############")
        loss_7_7 = dice_loss(logit_7x7, target_7x7)

        criterion = nn.BCEWithLogitsLoss()

        if self.cuda:
            criterion = criterion.to(f'cuda:{self.device_ids[0]}')

        bce_loss = criterion(logit.view(-1), target.view(-1))

        """ if self.batch_average:
            bce_loss /= n """

        return loss_5_5 + loss_7_7 + 2*bce_loss, loss_5_5+loss_7_7, bce_loss


if __name__=="__main__":
    TV_Loss = PatchTverskyLoss()
    B = 8
    C = 2
    H = 512
    W = 512
    logit = torch.randn(B, C, H, W)
    target = torch.empty(B, H, W, dtype=torch.long).random_(C)
    a, b, c = TV_Loss(logit, target)
    #print(a.size() , b.size(), c.size())