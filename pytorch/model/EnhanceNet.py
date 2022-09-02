import torch
import torch.nn as nn
from network.guided_filter import GuidedFilter
from .ModelZoo import DenseConvK3PRelu
from .HistogramLayer import DiffHistogramTransfer


class EnhanceNet(nn.Module):
    def __init__(self, in_ch, gf_kernel=5, int_layers=32, layers=4, ksize=3):
        super().__init__()
        self.in_ch = in_ch
        self.guided_filter = GuidedFilter(gf_kernel)
        self.hist_transfer = DiffHistogramTransfer(in_ch, int_layers, [1, ksize, ksize, 1], training=True,
                                                   activation='selu')
        self.post_transfer = DenseConvK3PRelu(2 * in_ch, int_layers, layers, 2 * in_ch, False)

    def forward(self, x):
        out = x - self.guided_filter(x, x)
        lum1 = torch.relu(self.hist_transfer(x))
        inp = torch.cat([lum1, out], dim=1)
        z = torch.relu(self.post_transfer(inp))
        return z[:, :self.in_ch] + z[:, self.in_ch:]
