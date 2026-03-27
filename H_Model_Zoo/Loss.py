import torch
import torch.nn as nn
import numpy as np


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, y, t):
        batch_size = y.shape[0]

        log_prob = torch.log_softmax(y, dim=1)
        loss = -log_prob[range(batch_size), t]

        acc_loss = torch.mean(loss)

        return acc_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.0, returnValue = None):
        super(FocalLoss, self).__init__()
        if alpha is not None:
            self.alpha = torch.tensor(alpha)  # alpha를 텐서로 변환
        else:
            self.alpha = None
        self.gamma = gamma
        self.returnValue = returnValue

    def forward(self, y, t):
        batch_size = y.shape[0]

        log_prob = torch.log_softmax(y, dim=1)
        prob = torch.softmax(y, dim=1)

        pt = prob[range(batch_size), t]

        if self.alpha is not None:
            loss = -self.alpha.to(t.device)[t] * (1 - pt) ** self.gamma * log_prob[range(batch_size), t]
        else:
            loss = -(1 - pt) ** self.gamma * log_prob[range(batch_size), t]

        if self.returnValue == None:
            acc_loss = loss
        else:
            acc_loss = torch.mean(loss)
        return acc_loss

