import torch
import numpy as np 

def iou_loss(y_true, y_pred, smooth = 1e-6, alpha = 0.7):
    y_true = y_true.to(torch.float32)
    y_pred = y_pred.to(torch.float32)
    
    y_true = y_true.view(y_true.size(0), -1)
    y_pred = y_pred.view(y_pred.size(0), -1)
    
    intersection = (y_true * y_pred).sum(dim=1)
    union = y_true.sum(dim=1) + y_pred.sum(dim=1) - intersection
    
    fg_weight = torch.mean(y_true, dim=1)
    bg_weight = 1 - fg_weight
    
    fg_intersection = (y_true * y_pred).sum(dim=1)
    fg_union = y_true.sum(dim=1) + y_pred.sum(dim=1) - fg_intersection
    
    bg_intersection = (1 - y_true) * (1 - y_pred).sum(dim=1)
    bg_union = (1 - y_true).sum(dim=1) + (1 - y_pred).sum(dim=1) - bg_intersection
    
    fg_iou = (fg_intersection + smooth) / (fg_union + smooth)
    bg_iou = (bg_intersection + smooth) / (bg_union + smooth)
    
    return fg_weight * fg_iou + bg_weight * bg_iou

def dice_coef_loss(y_true, y_pred, smooth = 1e-6):
    y_true = y_true.to(torch.float32)
    y_pred = y_pred.to(torch.float32)
    
    y_true = y_true.view(y_true.size(0), -1)
    y_pred = y_pred.view(y_pred.size(0), -1)
    
    intersection = (y_true * y_pred).sum(dim=1)
    
    dice = (2 * intersection + smooth) / (y_true.sum(dim=1) + y_pred.sum(dim=1) + smooth)
    
    return dice

def improved_combined_loss(y_true, y_pred):
    bce = torch.nn.BinaryCrossEntropyLoss()(y_true, y_pred)
    iou_loss_value = 1 - iou_loss(y_true, y_pred) 
    dice_loss_value = 1 - dice_coef_loss(y_true, y_pred)
    combined_loss = 0.4*bce + 0.3*iou_loss_value + 0.3*dice_loss_value
    
    return combined_loss
    