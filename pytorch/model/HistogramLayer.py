import torch
import torch.nn as nn
import network.netutils as mut


class HistogramLayer(nn.Module):
    def __init__(self, num_bins, minval=0, maxval=1, normalize=True, training=True, modulate=True):
        super(HistogramLayer, self).__init__()
        self.bins = num_bins
        self.normalized = normalize
        self.maxval = maxval
        self.minval = minval
        self.is_training = training
        self.modulate = modulate
        self._get_centers()
        self._get_widths()

    def _get_centers(self):
        init_bin_width = (self.maxval - self.minval) / self.bins
        bin_init_center = [(self.minval + k * init_bin_width + init_bin_width / 2) for k in range(self.bins)]
        centers = torch.FloatTensor(bin_init_center)
        if self.is_training:
            centers = nn.Parameter(centers)
            self.register_parameter('centers', centers)
        else:
            centers.requires_grad = False
            self.register_buffer('centers', centers)

    def _get_widths(self):
        diff = torch.cat([torch.FloatTensor((self.minval, )), self.centers[:-1]], dim=0)
        widths = self.centers - diff
        widths[0] = widths[0] * 2
        if self.is_training:
            widths = nn.Parameter(widths)
            self.register_parameter('widths', widths)
        else:
            widths.requires_grad = False
            self.register_buffer('widths', widths)

    def _rbf_kernel(self, x, centers, widths):
        return torch.exp(-1.0 * torch.square((1.0 / widths) * (x - centers))) + 1e-6

    def forward(self, x):
        in_ch = x.shape[1]
        centers = self.centers.repeat(in_ch).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        widths = self.widths.repeat(in_ch).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        x = x.repeat_interleave(self.bins, 1)
        out = self._rbf_kernel(x, centers, widths)
        if self.normalized:
            scale = torch.cat([torch.sum(out[:, (k * self.bins):((k + 1) * self.bins), ...], dim=1, keepdim=True)
                               for k in range(in_ch)], dim=1)
            scale = scale.repeat_interleave(self.bins, 1)
            out = out / scale
        if self.modulate:
            out = out * x
        return out


class DifferentialHistogram(nn.Module):
    def __init__(self, num_bins, minval=0, maxval=1, training=False):
        super(DifferentialHistogram, self).__init__()
        self.histogram = HistogramLayer(num_bins, minval, maxval, True, training, False)

    def forward(self, x):
        z = self.histogram(x)
        return torch.mean(z, dim=[2, 3], keepdim=False)


class DiffHistogramTransfer(nn.Module):
    def __init__(self, in_ch, num_bins, ksizes, minval=0, maxval=1, training=True, activation='selu'):
        super(DiffHistogramTransfer, self).__init__()
        self.in_ch = in_ch
        self.nbins = num_bins
        c = self.in_ch * self.nbins
        self.ksizes = ksizes
        self.layers = len(ksizes)
        trans = []
        for _ in range(self.layers):
            p = self.ksizes[_] // 2
            trans.append(nn.Sequential(nn.Conv2d(c, c, self.ksizes[_], 1, p, groups=c, bias=True),
                                       mut.get_act(activation)))
        self.trans = nn.Sequential(*trans)
        self.hist = HistogramLayer(num_bins, minval, maxval, True, training, True)

    def forward(self, x):
        h = self.hist(x)
        o = self.trans(h) + h
        o = torch.cat([torch.sum(o[:, (k * self.nbins):((k + 1) * self.nbins), ...], dim=1, keepdim=True)
                       for k in range(self.in_ch)], dim=1)
        return o
