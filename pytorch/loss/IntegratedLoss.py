import torch.nn as nn
import network.loss as loss


class IntegratedLoss(nn.Module):
    def __init__(self, loss_types, loss_weights):
        super(IntegratedLoss, self).__init__()
        if isinstance(loss_types, str):
            loss_types = [loss_types]
            loss_weights = [loss_weights]
        self.loss_weights = loss_weights
        self.items = len(loss_types)
        self.loss_collect = []
        self.desc = []
        self.loss_func = nn.ModuleList()
        for w in range(len(loss_types)):
            t = loss_types[w]
            if t == 'l1' or t == 'L1':
                ls = loss.ContentLoss('l1')
                desc = 'L1 loss'
            elif t == 'l2' or t == 'L2':
                ls = loss.ContentLoss('l1')
                desc = 'L2 loss'
            elif t == 'Color' or t == 'color':
                ls = loss.ColorLoss()
                desc = 'Color loss'
            elif t == 'VGG' or t == 'vgg':
                ls = loss.VGGLoss((0, 1, 2), None)
                desc = 'VGG loss'
            elif t == 'TVL_L1' or t == 'tvl_l1':
                ls = loss.TVLoss('l1')
                desc = 'L1 TV loss'
            elif t == 'TVL_L2' or t == 'tvl_l2':
                ls = loss.TVLoss('l2')
                desc = 'L2 TV loss'
            elif t == 'TVR_L1' or t == 'tvr_l1':
                ls = loss.TVRegulation('l1')
                desc = 'L1 TV regulation'
            elif t == 'TVR_L2' or t == 'tvr_l2':
                ls = loss.TVRegulation('l2')
                desc = 'L2 TV regulation'
            elif t == 'ssim' or t == 'SSIM':
                ls = loss.SSIMLoss()
                desc = 'SSIM loss'
            elif t == 'mssim' or t == 'MSSIM':
                ls = loss.MSSIMLoss()
                desc = 'MSSIM loss'
            elif t == 'Charbonnier' or t == 'charbonnier':
                ls = loss.CharbonnierLoss()
                desc = 'Charbonnier loss'
            else:
                raise ValueError('Unknown loss type: %s.' % t)
            self.loss_func.append(ls)
            self.desc.append(desc)

    def forward(self, x, y):
        loss_ = 0
        self.loss_collect.clear()
        for i in range(self.items):
            ls_ = self.loss_func[i](x, y)
            self.loss_collect.append(ls_)
            loss_ += self.loss_weights[i] * ls_
        return loss_

    def collect(self):
        return self.loss_collect
