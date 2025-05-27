import torch
import numpy as np
from torch import nn
from utils import check_target_range


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pridicts, tagets):
        pridicts = pridicts.view(pridicts.size(0), -1)
        tagets = tagets.view(tagets.size(0), -1)

        intersection = (pridicts * tagets).sum(dim=1)
        union = pridicts.sum(dim=1) + tagets.sum(dim=1)

        dsc = (2 * intersection + self.smooth) / (union + self.smooth)
        dice = 1 - dsc  # 1 - dice score
        return dice.mean()


class WIoULoss(torch.nn.Module):
    def __init__(self, alpha=0.7, smooth=1e-6):
        super(WIoULoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha

    def forward(self, predicts, tagets):
        predicts = predicts.view(tagets.size(0), -1)
        tagets = tagets.view(tagets.size(0), -1)

        fg_intersection = (predicts * tagets).sum(dim=1)
        fg_union = predicts.sum(dim=1) + tagets.sum(dim=1)
        iou_fg = (fg_intersection + self.smooth) / (
            fg_union - fg_intersection + self.smooth
        )

        bg_intersection = ((1 - predicts) * (1 - tagets)).sum(dim=1)
        bg_union = (1 - predicts).sum(dim=1) + (1 - tagets).sum(dim=1)
        iou_bg = (bg_intersection + self.smooth) / (
            bg_union - bg_intersection + self.smooth
        )

        wiou_loss = self.alpha * iou_fg + (1 - self.alpha) * iou_bg
        wiou_loss = 1 - wiou_loss
        return wiou_loss.mean()


class BCELoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super(BCELoss, self).__init__()
        self.smooth = smooth

    def forward(self, predicts, tagets):
        predicts = predicts.view(tagets.size(0), -1)
        tagets = tagets.view(tagets.size(0), -1)
        loss_log = tagets * torch.log(predicts)
        loss_log1 = (1 - tagets) * torch.log(1 - predicts)
        bce_loss = -(loss_log + loss_log1).sum(dim=1) / tagets.size(1)
        bce_loss = bce_loss.mean()
        return bce_loss


class CombinedLoss(torch.nn.Module):
    def __init__(self, alpha=0.7, smooth=1e-6, lamda=[0.4, 0.3, 0.3]):
        super(CombinedLoss, self).__init__()
        self.iou_loss = WIoULoss(alpha, smooth)
        self.dice_loss = DiceLoss(smooth)
        self.lamda = lamda
        self.bce = nn.BCELoss()

    def forward(self, predicts, tagets):
        if check_target_range(tagets):
            print("Target có giá trị ngoài khoảng [0, 1]")
            return torch.tensor(float('nan'))
        bce_loss_value = self.bce(predicts, tagets)
        iou_loss_value = self.iou_loss(predicts, tagets)
        dice_loss_value = self.dice_loss(predicts, tagets)
        # Check for NaN in any of the loss values
        # Kiểm tra xem mỗi loss có phải là NaN không
        if torch.isnan(bce_loss_value):
            print("BCE Loss có giá trị NaN")
            return torch.tensor(float('nan'))
            
        if torch.isnan(iou_loss_value):
            print("IOU Loss có giá trị NaN")
            return torch.tensor(float('nan'))
            
        if torch.isnan(dice_loss_value):
            print("Dice Loss có giá trị NaN")
            return torch.tensor(float('nan'))
        combined_loss = (
            self.lamda[0] * bce_loss_value
            + self.lamda[1] * iou_loss_value
            + self.lamda[2] * dice_loss_value
        )
        return combined_loss


# dice_loss = CombinedLoss()
# x = torch.randn(2, 3, 256, 256)
# y = torch.randn(2, 3, 256, 256)
# x = torch.sigmoid(x)  # Ensure predictions are in [0, 1]
# y = torch.sigmoid(y)  # Ensure targets are in [0, 1]
# loss = dice_loss(x, y)
# print(loss)  # Example output
