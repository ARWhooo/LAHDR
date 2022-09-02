import torch
import torch.nn as nn
import torch.nn.functional as F
import network.netutils as mut


class LinearChain(nn.Module):
    def __init__(self, in_ch, channels=None, layer_act='tanh', final_act='sigmoid', bias=True, norm=None):
        super().__init__()
        if channels is None:
            channels = [in_ch // 2, in_ch // 2, 2 * in_ch]
        self.in_ch = in_ch
        fc = []
        c = in_ch
        self.depth = len(channels)
        i = 0
        for l in channels:
            fc.append(mut.get_norm(norm, '1d')(c))
            fc.append(nn.Linear(c, l, bias=bias))
            c = l
            i += 1
            if i == self.depth:
                act = mut.get_act(final_act)
            else:
                act = mut.get_act(layer_act)
            fc.append(act)
        self.transfer = nn.Sequential(*fc)

    def forward(self, x):
        return self.transfer(x)


class SequentialConv2D(nn.Module):
    def __init__(self, in_ch,
                 layers,
                 ksizes=3,
                 strides=1,
                 paddings=1,
                 dilations=1,
                 groups=1,
                 bias=True,
                 layer_act='prelu',
                 final_act='prelu',
                 norm=None):
        super().__init__()
        self.in_ch = in_ch
        if not hasattr(layers, '__iter__'):
            layers = [layers]
        self.depth = len(layers)
        if not hasattr(ksizes, '__iter__'):
            ksizes = [ksizes] * self.depth
        if not hasattr(strides, '__iter__'):
            strides = [strides] * self.depth
        if not hasattr(paddings, '__iter__'):
            paddings = [paddings] * self.depth
        if not hasattr(dilations, '__iter__'):
            dilations = [dilations] * self.depth
        if not hasattr(groups, '__iter__'):
            groups = [groups] * self.depth
        transfer = []
        c = in_ch
        for i in range(self.depth):
            transfer.append(mut.get_norm(norm)(c))
            transfer.append(nn.Conv2d(c, layers[i], ksizes[i], strides[i], paddings[i], dilations[i], groups[i], bias))
            c = layers[i]
            if i == self.depth - 1:
                act = mut.get_act(final_act)
            else:
                act = mut.get_act(layer_act)
            transfer.append(act)
        self.transfer = nn.Sequential(*transfer)

    def forward(self, x):
        return self.transfer(x)


class DenseBlock(nn.Module):
    def __init__(self, bottleneck, in_ch, lat_ch, layers, out_ch):
        super(DenseBlock, self).__init__()
        self.layers = layers
        self.in_ch = in_ch
        self.lat_ch = lat_ch
        self.out_ch = out_ch
        ch = in_ch
        for i in range(self.layers):
            setattr(self, 'transfer%d' % (i + 1), bottleneck(ch, self.lat_ch))
            ch += lat_ch
        self.final = bottleneck(ch, out_ch)

    def forward(self, x):
        t = x
        for i in range(self.layers):
            t = torch.cat([t, getattr(self, 'transfer%d' % (i + 1))(t)], dim=1)
        return self.final(t)


class DenseConvK3PRelu(nn.Module):
    def __init__(self, in_ch, lat_ch, layers, out_ch, bias=True, norm='none'):
        super(DenseConvK3PRelu, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.lat_ch = lat_ch
        self.layers = layers
        self.bottleneck = lambda i, o: nn.Sequential(*[mut.get_norm(norm)(i),
                                                       nn.Conv2d(i, o, 3, 1, 1, bias=bias, padding_mode='reflect'),
                                                       nn.PReLU()])
        self.transfer = DenseBlock(self.bottleneck, self.in_ch, self.lat_ch, self.layers, self.out_ch)

    def forward(self, x):
        return self.transfer(x)


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, ksize, stride, padding, bias=False, act='relu', norm='', padding_mode='reflect'):
        super(ConvLayer, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.ksize = ksize
        self.s = stride
        self.padding = padding
        self.bias = bias
        self.mode = padding_mode
        self.conv = nn.Conv2d(in_ch, out_ch, ksize, stride, padding, bias=bias, padding_mode=padding_mode)
        self.activation = mut.get_act(act)
        self.norm = mut.get_norm(norm)(in_ch)

    def forward(self, x):
        return self.activation(self.conv(self.norm(x)))


class SpatialAttention2D(nn.Module):
    def __init__(self):
        super(SpatialAttention2D, self).__init__()
        self.trans = ConvLayer(2, 1, 3, 1, 1, True, 'sigmoid')

    def forward(self, x):
        max_ = torch.max(x, dim=1).values
        mean_ = torch.mean(x, dim=1)
        return x * self.trans(torch.cat([max_, mean_], dim=1))


class ChannelAttention2D(nn.Module):
    def __init__(self, channels, reduction=4):
        super(ChannelAttention2D, self).__init__()
        self.channels = channels
        self.reduce = reduction
        c = channels * 2
        r = reduction
        self.trans = nn.Sequential(*[nn.Linear(c, c // r), nn.Tanh(), nn.Linear(c // r, c), nn.Sigmoid()])

    def forward(self, x):
        l = torch.mean(x, dim=[2, 3], keepdim=False)
        m = torch.max(torch.max(x, dim=3, keepdim=False).values, dim=2, keepdim=False).values
        L = torch.unsqueeze(torch.unsqueeze(self.trans(torch.cat([l, m], dim=1)), dim=-1), dim=-1)
        L1 = L[:, :self.channels]
        L2 = L[:, self.channels:]
        return 0.5 * (L1 * x + L2 * x)

