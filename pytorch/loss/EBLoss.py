import torch
import torch.nn as nn
import network.loss as loss


class EBLoss(nn.Module):
    def __init__(self, loss_type='mean', norm='l1'):
        super().__init__()
        self.eps = 1e-6
        id = ''
        if loss_type == 'mean':
            self.transfer = self._mean
            id += 'Mean '
        else:
            self.transfer = self._square_log_mean
            id += 'Bright level '
        self.metric = loss.ContentLoss(norm)
        if norm == 'l1':
            id += 'L1 loss'
        else:
            id += 'L2 loss'
        self.desc = [id]
        self.loss_collect = []

    def _mean(self, x):
        return torch.mean(x, dim=[1, 2, 3])

    def _square_log_mean(self, x):
        return torch.square(torch.mean(torch.log(x + self.eps), dim=[1, 2, 3]))

    def forward(self, x, y):
        self.loss_collect.clear()
        ls = self.metric(self.transfer(x), self.transfer(y))
        self.loss_collect.append(ls)
        return ls

    def collect(self):
        return self.loss_collect
