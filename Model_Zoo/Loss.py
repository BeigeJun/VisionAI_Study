import torch
import torch.nn as nn
import numpy as np


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, score, target):
        batch_size = len(score)
        loss = torch.log(torch.sum(torch.exp(score), dim=1)) - score[range(batch_size), target]
        acc_loss = torch.mean(loss)

        return acc_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, score, target):
        batch_size = len(score)

        softmax_score = torch.softmax(score, dim=1)

        target_prob = softmax_score[range(batch_size), target]

        if self.alpha is not None:
            loss = -self.alpha[target] * (1 - target_prob) ** self.gamma * torch.log(target_prob)
        else:
            loss = - (1 - target_prob) ** self.gamma * torch.log(target_prob)

        if self.reduction == 'mean':
            acc_loss = torch.mean(loss)
        elif self.reduction == 'sum':
            acc_loss = torch.sum(loss)
        else:
            acc_loss = loss

        return acc_loss
