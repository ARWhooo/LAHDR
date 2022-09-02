import torch.nn as nn
from network.guided_filter import GuidedFilter
from loss import IntegratedLoss


class LogLoss(nn.Module):
    def __init__(self, enables=None, loss_types=None, loss_weights=None, eps=1e-6):
        super(LogLoss, self).__init__()
        self.loss_handles = IntegratedLoss(loss_types, loss_weights)
        self.enables = enables
        self.length = self.loss_handles.items
        self.desc = ['Log ' + self.loss_handles.desc[i] if enables[i]
                     else self.loss_handles.desc[i] for i in range(self.length)]
        self.loss_collect = []
        self.eps = eps

    def _transfer(self, x):
        return (x + self.eps).log()

    def forward(self, x, y):
        x_ = self._transfer(x)
        y_ = self._transfer(y)
        loss_ = 0
        self.loss_collect.clear()
        for i in range(self.length):
            if self.enables[i]:
                ls_ = self.loss_handles.loss_func[i](x_, y_)
            else:
                ls_ = self.loss_handles.loss_func[i](x, y)
            self.loss_collect.append(ls_)
            loss_ += self.loss_handles.loss_weights[i] * ls_
        return loss_

    def collect(self):
        return self.loss_collect
