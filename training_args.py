import torch
import torch.nn as nn
import torch.nn.functional as F

def Make_Optimizer(model):
    # 논문 설정: Adam optimizer with learning rate 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return optimizer

def Make_LR_Scheduler(optimizer):
    # CosineAnnealingLR with large enough T_max (예: 50 epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=30,       # 주기 길이: 학습 epoch 수에 맞춰 조절
        eta_min=1e-4    # 최소 learning rate
    )
    return scheduler

def Make_Loss_Function(number_of_classes):
    if number_of_classes == 1:
        return nn.BCEWithLogitsLoss()
    else:
        return nn.CrossEntropyLoss()

def Make_Loss_Function(number_of_classes):
    class DiceCELoss:
        def __init__(self, weight=0.5, epsilon=1e-6, mode='multiclass'):
            self.weight = weight
            self.epsilon = epsilon
            self.mode = mode
        
        def __call__(self, pred, target):
            if self.mode == 'binary':
                pred = pred.squeeze(1)  # shape: (batchsize, H, W)
                target = target.squeeze(1).float()
                intersection = torch.sum(pred * target, dim=(1, 2))
                union = torch.sum(pred, dim=(1, 2)) + torch.sum(target, dim=(1, 2))
                dice = (2 * intersection + self.epsilon) / (union + self.epsilon)
                dice_loss = 1 - dice.mean()
                
                ce_loss = F.binary_cross_entropy(pred, target)
            
            elif self.mode == 'multiclass':
                batchsize, num_classes, H, W = pred.shape
                target = target.squeeze(1)
                target_one_hot = F.one_hot(target, num_classes=num_classes).squeeze(1).permute(0, 3, 1, 2).float()
                intersection = torch.sum(pred * target_one_hot, dim=(2, 3))
                union = torch.sum(pred, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))
                dice = (2 * intersection + self.epsilon) / (union + self.epsilon)
                dice_loss = 1 - dice.mean()
                
                ce_loss = F.cross_entropy(pred, target)
            else:
                raise ValueError("mode should be 'binary' or 'multiclass'")
            
            combined_loss = self.weight * dice_loss + (1 - self.weight) * ce_loss
            
            return combined_loss
    
    BINARY_SEG = True if number_of_classes==2 else False
    return DiceCELoss(mode='binary') if BINARY_SEG else DiceCELoss(mode='multiclass') 
    

