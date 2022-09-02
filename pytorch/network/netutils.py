import torch
from torch import nn


class Gradient(nn.Module):
    def __init__(self, direction=None, norm='l1'):
        super(Gradient, self).__init__()
        if direction is None:
            self.dir = 'both'
        else:
            self.dir = direction
        self.norm = norm
        self.pad_x = nn.ReflectionPad2d((0, 1, 0, 0))
        self.pad_y = nn.ReflectionPad2d((0, 0, 0, 1))

    def _grad_x(self, x):
        x = x[..., :-1] - x[..., 1:]
        return self.pad_x(x)

    def _grad_y(self, x):
        x = x[..., :-1, :] - x[..., 1:, :]
        return self.pad_y(x)

    def forward(self, x):
        if self.dir == 'x':
            x_ = self._grad_x(x)
            y_ = torch.zeros_like(x_)
        elif self.dir == 'y':
            y_ = self._grad_y(x)
            x_ = torch.zeros_like(y_)
        else:
            x_ = self._grad_x(x)
            y_ = self._grad_y(x)
        if self.norm == 'l1' or self.norm == 'L1':
            return 0.5 * (x_.abs() + y_.abs())
        else:
            return torch.sqrt(x_.square() + y_.square())


# from github repo: deep-image-prior
class GenNoise(nn.Module):
    def __init__(self, dim2):
        super(GenNoise, self).__init__()
        self.dim2 = dim2

    def forward(self, input):
        a = list(input.size())
        a[1] = self.dim2
        # print (input.data.type())
        b = torch.zeros(a).type_as(input.data)
        b.normal_()
        x = torch.autograd.Variable(b)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, ksize):
        super(SpatialAttention, self).__init__()
        self.ksize = ksize
        self.conv = nn.Sequential(*[nn.Conv2d(3, 1, ksize, 1, ksize // 2, bias=True, padding_mode='reflect'),
                                    nn.Sigmoid()])

    def forward(self, x):
        a = torch.mean(x, dim=1, keepdim=True)
        b = torch.min(x, dim=1, keepdim=True).values
        c = torch.max(x, dim=1, keepdim=True).values
        z = torch.cat([a, b, c], dim=1)
        return self.conv(z) * x


class ChannelAttention(nn.Module):
    def __init__(self, in_ch, reduction):
        super(ChannelAttention, self).__init__()
        self.reduce_ratio = reduction
        self.fc1 = nn.Linear(in_ch, in_ch // reduction, bias=True)
        self.fc2 = nn.Linear(in_ch // reduction, in_ch, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        l = x.mean(dim=3).mean(dim=2)
        z = self.act(self.fc2(self.act(self.fc1(l))))
        return x * z.unsqueeze(dim=-1).unsqueeze(dim=-1)


class DualAttention(nn.Module):
    def __init__(self, in_ch, ksize, reduction):
        super(DualAttention, self).__init__()
        self.ksize = ksize
        self.conv = nn.Sequential(*[nn.Conv2d(3, 1, ksize, 1, ksize // 2, bias=True, padding_mode='reflect'),
                                    nn.Sigmoid()])
        self.reduce_ratio = reduction
        self.fc1 = nn.Linear(in_ch, in_ch // reduction, bias=True)
        self.fc2 = nn.Linear(in_ch // reduction, in_ch, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        a = torch.mean(x, dim=1, keepdim=True)
        b = torch.min(x, dim=1, keepdim=True).values
        c = torch.max(x, dim=1, keepdim=True).values
        s = torch.cat([a, b, c], dim=1)
        l = torch.mean(x, dim=[2, 3], keepdim=False)
        z1 = self.act(self.fc2(self.act(self.fc1(l))))
        z2 = self.conv(s)
        o = x * z1.unsqueeze(dim=-1).unsqueeze(dim=-1)
        return o * z2


# from github repo: deep-image-prior
class swish(nn.Module):
    """
        https://arxiv.org/abs/1710.05941
        The hype was so huge that I could not help but try it
    """
    def __init__(self):
        super(swish, self).__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return x * self.s(x)


def get_act(desc):
    if desc is None or desc == 'none' or desc == '':
        return nn.Identity()
    elif desc == 'sigmoid' or desc == 'SIGMOID':
        return nn.Sigmoid()
    elif desc == 'relu' or desc == 'RELU' or desc == 'ReLU':
        return nn.ReLU()
    elif desc == 'prelu' or desc == 'leaky-relu' or desc == 'PReLU' or desc == 'PRELU':
        return nn.PReLU()
    elif desc == 'tanh' or desc == 'Tanh' or desc == 'TANH':
        return nn.Tanh()
    elif desc == 'selu' or desc == 'SELU':
        return nn.SELU()
    elif desc == 'elu' or desc == 'ELU':
        return nn.ELU()
    elif desc == 'swish' or desc == 'SWISH':
        return swish()
    elif desc == 'gelu' or desc == 'GELU':
        return nn.GELU()
    else:
        raise RuntimeError('Undefined activation type: %s' % desc)


def get_norm(desc, dimen='2d'):
    if dimen == '2d' or dimen == '2D':
        BN = nn.BatchNorm2d
        IN = nn.InstanceNorm2d
    elif dimen == '3d' or dimen == '3D':
        BN = nn.BatchNorm3d
        IN = nn.InstanceNorm3d
    else:
        BN = nn.BatchNorm1d
        IN = nn.InstanceNorm1d
    if desc is None or desc == 'none' or desc == '':
        return nn.Identity
    elif desc == 'BatchNorm' or desc == 'batchnorm' or desc == 'bn' or desc == 'BN':
        return BN
    elif desc == 'InstanceNorm' or desc == 'instancenorm' or desc == 'in' or desc == 'IN':
        return IN
    elif desc == 'LayerNorm' or desc == 'layernorm' or desc == 'ln' or desc == 'LN':
        return nn.LayerNorm
    else:
        raise RuntimeError('Undefined normalization type: %s' % desc)


# Extracted from mmcv project
def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)
