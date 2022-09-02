import torch.nn as nn
from utils.algorithm import mu_law_inverse
from loss import IntegratedLoss


class MULawLoss(nn.Module):
    def __init__(self, mu=5000, mu_enables=None, loss_types=None, loss_weights=None):
        super(MULawLoss, self).__init__()
        self.mu = mu
        self.loss_handles = IntegratedLoss(loss_types, loss_weights)
        self.mu_enables = mu_enables
        self.length = self.loss_handles.items
        self.desc = ['MU%d ' % mu + self.loss_handles.desc[i] if mu_enables[i] else self.loss_handles.desc[i]
                     for i in range(self.length)]
        self.loss_collect = []

    def forward(self, x, y):
        x_ = mu_law_inverse(x, self.mu)
        y_ = mu_law_inverse(y, self.mu)
        loss_ = 0
        self.loss_collect.clear()
        for i in range(self.length):
            if self.mu_enables[i]:
                ls_ = self.loss_handles.loss_func[i](x_, y_)
            else:
                ls_ = self.loss_handles.loss_func[i](x, y)
            self.loss_collect.append(ls_)
            loss_ += self.loss_handles.loss_weights[i] * ls_
        return loss_

    def collect(self):
        return self.loss_collect
