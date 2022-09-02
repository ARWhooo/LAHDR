import torch
import torch.nn as nn
from network.netutils import get_act
from model.ModelZoo import LinearChain, ConvLayer
from model.HistogramLayer import DifferentialHistogram
import utils.algorithm as alg


class GlobalLightRectification(object):
    def __init__(self, mu=200.0, factor=3.0, norm_enable=False):
        self.multi_factor = factor
        self.mu_value = mu
        self.norm_enable = norm_enable

    def __call__(self, inp, eb):
        eb = eb.unsqueeze(-1).unsqueeze(-1) * self.multi_factor
        x = alg.mu_law_forward(inp, mu=self.mu_value)
        x = x * (2 ** (-1.0 * eb))
        x = alg.mu_law_inverse(x, mu=self.mu_value)
        if self.norm_enable:
            x = x / torch.max(x)
        else:
            x = torch.clamp(x, 0, 1)
        return x, eb


class MultiConv(nn.Module):
    def __init__(self, in_ch, out_ch, ksize1=3, ksize2=5, stride=2, bias=False, act='prelu'):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, ksize1, stride, (ksize1 - 1) // 2, bias=bias,
                                             padding_mode='reflect'),
                                   get_act(act))
        self.conv2 = nn.Sequential(nn.Conv2d(in_ch, out_ch, ksize2, stride, (ksize2 - 1) // 2, bias=bias,
                                             padding_mode='reflect'),
                                   get_act(act))

    def forward(self, x):
        return torch.cat([self.conv1(x), self.conv2(x)], dim=1)


class EBNet(nn.Module):
    def __init__(self, int_layers=32, hist_layers=64):
        super().__init__()
        self.int_layers = int_layers
        self.hist_channels = hist_layers
        # Global branch
        self.histogram = DifferentialHistogram(hist_layers, training=False)
        self.hist_fc = nn.Sequential(nn.Linear(2 * hist_layers, 2 * hist_layers, bias=True), nn.Tanh())
        # Local branch
        self.spconv1 = MultiConv(3, int_layers, 3, 5, 2, False, 'prelu')
        self.spconv2 = nn.Sequential(nn.Conv2d(2 * int_layers, 2 * int_layers, 3, 2, 1, bias=False,
                                               padding_mode='reflect'), nn.PReLU())
        self.spconv3 = nn.Sequential(nn.Conv2d(2 * int_layers, 4 * int_layers, 3, 2, 1, bias=False,
                                               padding_mode='reflect'), nn.PReLU())
        self.spconv4 = nn.Sequential(nn.Conv2d(4 * int_layers, 8 * int_layers, 3, 2, 1, bias=False,
                                               padding_mode='reflect'), nn.PReLU())
        self.spnorm = nn.BatchNorm2d(8 * int_layers)
        # Branches Merging
        self.final = nn.Sequential(nn.Linear(2 * hist_layers + 8 * int_layers, 120, bias=True),
                                   nn.Tanh(),
                                   nn.Linear(120, 20, bias=True),
                                   nn.Tanh(),
                                   nn.Linear(20, 1, bias=True),
                                   nn.Tanh())

    def forward(self, x):
        assert x.size()[1] == 3, "Error: input tensor must have exact 3 channels, but got %d instead." % x.size()[1]
        xmax = torch.max(x, dim=1, keepdim=True).values
        xsat = alg.get_saturation(x)
        hist_inp = torch.cat([xmax, xsat], dim=1)
        hist_info = self.hist_fc(self.histogram(hist_inp))
        spat = self.spconv4(self.spconv3(self.spconv2(self.spconv1(x))))
        spat_info = torch.mean(self.spnorm(spat), dim=[2, 3], keepdim=False)
        info = torch.cat([spat_info, hist_info], dim=1)
        return self.final(info)
