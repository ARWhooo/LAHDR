import torch
import torch.nn as nn
from .ModelZoo import ConvLayer
from network.netutils import get_act, get_norm


class MultiDilatedConv(nn.Module):
    def __init__(self, in_ch, lat_ch, out_ch, ksize, dilates, bias=True, act='prelu', norm='none'):
        super().__init__()
        self.dilates = dilates
        for d in dilates:
            padding = (ksize + 2 * (d - 1)) // 2
            setattr(self, 'dil_conv%d' % d, nn.Sequential(nn.Conv2d(in_ch, lat_ch, ksize, 1, 0, d, bias=bias,
                                                                    padding_mode='reflect'),
                                                          nn.ReflectionPad2d(padding),
                                                          get_act(act)))
        self.norm = get_norm(norm)(len(dilates) * lat_ch)
        self.merge_conv = ConvLayer(len(dilates) * lat_ch, out_ch, 1, 1, 0, bias=bias, act=act)
        self.skip_conv = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)

    def forward(self, x):
        branch = torch.cat([getattr(self, 'dil_conv%d' % d)(x) for d in self.dilates], dim=1)
        branch = self.merge_conv(self.norm(branch))
        return branch + self.skip_conv(x)


class DeNet(nn.Module):
    def __init__(self, in_ch, out_ch, int_layers=32, layers=5, ksize=3):
        super().__init__()
        self.int_layers = int_layers
        self.layers = layers
        self.ksize = ksize
        transfer = lambda i, o: MultiDilatedConv(i, o, o, ksize, [d for d in range(1, ksize + 1)], bias=False,
                                                 act='prelu')
        ch = in_ch
        for i in range(self.layers - 1):
            setattr(self, 'transfer%d' % (i + 1), transfer(ch, self.lat_ch))
            ch += int_layers
        self.final = ConvLayer(ch, out_ch, ksize, 1, (ksize - 1) // 2, bias=False, act='tanh')
        if in_ch != out_ch:
            self.skip_conv = nn.Conv2d(in_ch, out_ch, 1, 1, 0)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x):
        t = x
        for i in range(self.layers - 1):
            t = torch.cat([t, getattr(self, 'transfer%d' % (i + 1))(t)], dim=1)
        out = self.final(t)
        return (self.skip_conv(x) + out) / 2.0
