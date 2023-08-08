import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import resnet


class SegmentationLosses(nn.Module):
    def __init__(self, cuda=True):
        # self.ignore_index = ignore_index
        # self.weight = weight
        # self.size_average = size_average
        self.cuda = cuda

    def forward(self, logit, target):
        n, c, h, w = logit.size()
        smooth = 1
        probas = torch.sigmoid(logit)
        true_1_hot = target
        
        logit_5x5 = probas.unfold(2,5,5).unfold(3,5,5).contiguous()
        #print('logit 5x5: ',logit_5x5.size())
        logit_5x5 = logit_5x5.view(n, -1, 5, 5)
        #print('logit 5x5 flatten: ',logit_5x5.size())
        logit_7x7 = probas.unfold(2,7,7).unfold(3,7,7).contiguous()
        #print('logit 7x7: ',logit_7x7.size())
        logit_7x7 = logit_7x7.view(n, -1, 7, 7)
        #print('logit 7x7 flatten: ',logit_7x7.size())

        target_5x5 = true_1_hot.unfold(2,5,5).unfold(3,5,5).contiguous()
        target_5x5 = target_5x5.view(n, -1, 5, 5)
        target_7x7 = true_1_hot.unfold(2,7,7).unfold(3,7,7).contiguous()
        target_7x7 = target_7x7.view(n, -1, 7, 7)

        # pdb.set_trace()

        def dice_loss(current_logit, current_target, smooth=smooth):
            dims = (0,) + tuple(range(2, current_target.ndimension()))
            #dims = (1,2,3)
            #print('dims: ',dims)
            intersection = torch.sum(current_logit * current_target, dims)
            #print('intersection: ',intersection.size())
            cardinality = torch.sum(current_logit + current_target, dims)
            #print('cardinality: ',cardinality.size())
            dice_Loss = ((2. * intersection + smooth) / (cardinality + smooth)).mean()
            return 1-dice_Loss


        # loss_3_3 = dice_loss(logit_3x3, target_3x3)
        # pdb.set_trace()
        loss_5_5 = dice_loss(logit_5x5, target_5x5)
        loss_7_7 = dice_loss(logit_7x7, target_7x7)

        criterion = nn.BCEWithLogitsLoss()

        if self.cuda:
            criterion = criterion.cuda()

        ce_loss = criterion(logit.view(-1), target.view(-1))

        """ if self.batch_average:
            ce_loss /= n """

        #print('ce loss: ',ce_loss)

        return loss_5_5 + loss_7_7 + 2*ce_loss


if __name__ == "__main__":
    a = torch.rand(4, 1, 512, 512).cuda()
    b = torch.rand(4, 1, 512, 512).cuda()
    #print(DiceLoss(a, b).item())
    # #print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    # #print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())