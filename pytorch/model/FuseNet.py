import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .ModelZoo import SequentialConv2D, ConvLayer
import utils.algorithm as alg


def get_multi_images(inp1, gamma1=1.5, gamma2=1.5):
    inp2 = 1 - torch.pow(1 - torch.pow(inp1, 1 / gamma1), gamma1)
    inp3 = torch.pow(1 - torch.pow(1 - inp1, 1 / gamma2), gamma2)
    return inp1, inp2, inp3


class EncoderConv(nn.Module):
    def __init__(self, in_ch, ksize, layers=None, downscales=None, maxdown=False, bias=False, act='prelu',
                 norm='none'):
        super().__init__()
        if layers is None:
            self.layers = [16, 64, 256]
        else:
            self.layers = layers
        self.scales = len(self.layers)
        if downscales is None:
            self.downscales = [2] * self.scales
        else:
            self.downscales = downscales
        for i in range(self.scales):
            if i == 0:
                z = ConvLayer(in_ch, layers[i], ksize, 1, (ksize - 1) // 2, bias, act, norm)
            else:
                z = ConvLayer(layers[i - 1], layers[i], ksize, 1, (ksize - 1) // 2, bias, act, norm)
            setattr(self, 'transfer%d' % (i + 1), z)
        for i in range(self.scales):
            if maxdown:
                setattr(self, 'down%d' % (i + 1), nn.MaxPool2d(self.downscales[i], self.downscales[i]))
            else:
                k = self.downscales[i] + 1
                d = self.downscales[i]
                p = (k - 1) // 2
                setattr(self, 'down%d' % (i + 1), nn.Conv2d(layers[i], layers[i], k, d, p, bias=False))

    def forward(self, x):
        out = OrderedDict()
        shapes = OrderedDict()
        z = x
        for i in range(self.scales):
            z = getattr(self, 'transfer%d' % (i + 1))(z)
            h, w = z.shape[-2:]
            shapes[i + 1] = (h, w)
            out[i + 1] = z
            z = getattr(self, 'down%d' % (i + 1))(z)
        return z, out, shapes


class FusionMask(nn.Module):
    def __init__(self, trainable=False):
        super().__init__()
        self.sigma = 0.2
        if trainable:
            self.trainable = True
            mu = torch.ones((1, )) * 0.5
            mu.requires_grad = True
            self.register_parameter('mu', nn.Parameter(mu))
        else:
            self.trainable = False
            mu = torch.ones((1,)) * 0.5
            mu.requires_grad = False
            self.register_buffer('mu', nn.Parameter(mu))
        if trainable:
            self.transfer = SequentialConv2D(3, [3, 1], [1, 1], paddings=[0, 0], bias=False,
                                             layer_act='none', final_act='sigmoid')
        else:
            self.transfer = self._transfer

    def _transfer(self, x):
        return x[:, 0:1] * x[:, 1:2] * x[:, 2:3]

    def _ev_level(self, x):
        dem = 2 * self.sigma * self.sigma
        out = -1.0 * torch.square(x - self.mu)
        return torch.exp(out / dem)

    def forward(self, x):
        lum = alg.get_luminance(x).unsqueeze(1)
        t = alg.get_gradient(lum)
        s = alg.get_saturation(x)
        e = self._ev_level(lum)
        return self.transfer(torch.cat([t, s, e], dim=1))


class FeatureMasksGeneration(nn.Module):
    def __init__(self, heads, shared=True, trainable=False):
        super().__init__()
        if shared:
            fm = FusionMask(trainable)
            self.fmg = nn.ModuleList([fm] * heads)
        else:
            fmg = []
            for _ in range(heads):
                fmg.append(FusionMask(trainable))
            self.fmg = nn.ModuleList(fmg)
        self.heads = heads
        self.shared = shared
        self.trainable = trainable
        self.maskconv = nn.Conv2d(heads, heads, 3, 1, 1, 1, groups=heads, bias=True, padding_mode='reflect')
        self.out = nn.Softmax(dim=1)

    def forward(self, images):
        fms = [self.fmg[i](images[i]) for i in range(self.heads)]
        out = self.out(self.maskconv(torch.cat(fms, dim=1)))
        return torch.split(out, [1] * self.heads, dim=1)


class GlobalFeatureEmbedding(nn.Module):
    def __init__(self, heads, in_ch, base_ch, depth, out_ch, seperate=False, use_bias=False, lat_norm=True):
        super().__init__()
        self.heads = heads
        self.base_ch = base_ch
        self.depth = depth
        en_chs = []
        for _ in range(depth):
            en_chs.append(base_ch)
            base_ch *= 2
        if not seperate:
            self.encoders = nn.ModuleList([EncoderConv(in_ch, 3, en_chs, [2] * depth, True, use_bias, 'prelu')] * heads)
        else:
            enc = []
            for _ in range(heads):
                enc.append(EncoderConv(in_ch, 3, en_chs, [2] * depth, True, use_bias, 'prelu'))
            self.encoders = nn.ModuleList(enc)
        self.down = nn.MaxPool2d(2)
        norms = []
        for _ in range(heads):
            if lat_norm:
                norms.append(nn.BatchNorm2d(en_chs[-1]))
            else:
                norms.append(nn.Identity())
        self.norms = nn.ModuleList(norms)
        self.out_fc = nn.Sequential(nn.Linear(heads * en_chs[-1], out_ch, True), nn.Sigmoid())

    def forward(self, feats, masks):
        latents = []
        skips = []
        feats_ = [F.interpolate(f, (128, 128), mode='bilinear', align_corners=True) for f in feats]
        masks_ = [F.interpolate(m, (128, 128), mode='bilinear', align_corners=True) for m in masks]
        for i in range(self.heads):
            lat, skip, shapes = self.encoders[i](feats_[i])
            for j in range(self.depth):
                masks_[j] = self.down(masks_[j])
            latents.append(lat)
            skips.append(skip)
        latents = [self.norms[_](l) for _, l in zip(range(self.heads), latents)]
        glbs = [torch.mean(la * msk, dim=[2, 3], keepdim=False) for la, msk in zip(latents, masks_)]
        return self.out_fc(torch.cat(glbs, dim=1))


class FuseNet(nn.Module):
    def __init__(self, heads, in_ch, base_ch, depth, seperate=False, mode='concat', bias=False, act='prelu'):
        super().__init__()
        self.in_ch = in_ch
        self.base_ch = base_ch
        self.seperate = seperate
        self.heads = heads
        self.depth = depth
        self.mode = mode
        self.bias = bias
        self.act = act
        self.down = nn.MaxPool2d(2)
        enc_chs = []
        for _ in range(depth):
            enc_chs.append(base_ch)
            base_ch *= 2
        if not seperate:
            self.encoders = nn.ModuleList([EncoderConv(in_ch, 3, enc_chs, None, False, bias, act)] * heads)
        else:
            enc = []
            for _ in range(heads):
                enc.append(EncoderConv(in_ch, 3, enc_chs, None, False, bias, act))
            self.encoders = nn.ModuleList(enc)
        self.FMG = FeatureMasksGeneration(heads, shared=False, trainable=True)
        self.GFE = GlobalFeatureEmbedding(heads, in_ch, base_ch, depth, enc_chs[-1], seperate=False,
                                          use_bias=bias, lat_norm=True)
        self._init_latent_layer(enc_chs[-1], enc_chs[-1])
        enc_chs.append(enc_chs[-1])
        for i in range(depth, 0, -1):
            self._init_decoder_layer(i, enc_chs[i], enc_chs[i - 1], enc_chs[i - 1])
        self.final = ConvLayer(enc_chs[0], in_ch, 3, 1, 1, bias=self.bias, act='relu')

    def _init_latent_layer(self, in_ch, out_ch):
        self.latconv1 = SequentialConv2D(in_ch, [in_ch, in_ch], 3, 1, 1, bias=self.bias, layer_act=self.act,
                                         final_act=self.act)
        self.latconv2 = SequentialConv2D(in_ch * 2, [out_ch, out_ch], 3, 1, 1, bias=self.bias, layer_act=self.act,
                                         final_act=self.act)

    def _exec_latent_layer(self, latents, masks, globals):
        lat = sum([l * m for l, m in zip(latents, masks)])
        lat = self.latconv1(lat)
        _, C, H, W = lat.size()
        globals = globals.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
        lat = torch.cat([lat, globals], dim=1)
        return self.latconv2(lat)

    def _init_decoder_layer(self, layer, in_ch, skip_ch, out_ch):
        if self.mode == 'concat':
            setattr(self, 'upconv%d_1' % layer, ConvLayer(in_ch, out_ch, 3, 1, 1, bias=self.bias, act=self.act))
            setattr(self, 'upconv%d_2' % layer, ConvLayer(skip_ch + out_ch, out_ch, 3, 1, 1, bias=self.bias,
                                                          act=self.act))
        else:
            setattr(self, 'upconv%d_1' % layer, ConvLayer(in_ch, out_ch, 3, 1, 1, bias=self.bias, act=self.act))
            setattr(self, 'upconv%d_2' % layer, ConvLayer(out_ch, out_ch, 3, 1, 1, bias=self.bias,
                                                          act=self.act))

    def _exec_decoder_layer(self, layer, lat, skips, masks):
        skip = 0
        for _ in range(self.heads):
            skip += skips[_][layer] * masks[_][layer]
        if self.mode == 'concat':
            lat = getattr(self, 'upconv%d_1' % layer)(lat)
            lat = torch.cat([lat, skip], dim=1)
            lat = getattr(self, 'upconv%d_2' % layer)(lat)
        elif self.mode == 'mid_add':
            lat = getattr(self, 'upconv%d_1' % layer)(lat)
            lat = lat + skip
            lat = getattr(self, 'upconv%d_2' % layer)(lat)
        else:
            lat = getattr(self, 'upconv%d_1' % layer)(lat)
            lat = getattr(self, 'upconv%d_2' % layer)(lat)
            lat += skip
        return lat

    def _generate_masks_stack(self, masks):
        masks_stack = OrderedDict()
        for _ in range(self.heads):
            m_ = masks[_]
            masks_stack[_] = OrderedDict()
            id = 1
            for i in range(self.depth):
                masks_stack[_][id] = m_
                m_ = self.down(m_)
                id += 1
            masks_stack[_][id] = m_
        return masks_stack

    def forward(self, images):
        masks = self.FMG(images)
        glbs = self.GFE(images, masks)
        skips_stack = []
        latents = []
        shapes = OrderedDict()
        m_stack = self._generate_masks_stack(masks)
        for _ in range(self.heads):
            lat, skips, shapes = self.encoders[_](images[_])
            latents.append(lat)
            skips_stack.append(skips)
        masks_ = [m_stack[_][self.depth + 1] for _ in range(self.heads)]
        lat = self._exec_latent_layer(latents, masks_, glbs)
        for j in range(self.depth, 0, -1):
            lat = F.interpolate(lat, shapes[j], mode='bilinear', align_corners=True)
            lat = self._exec_decoder_layer(j, lat, skips_stack, m_stack)
        return self.final(lat)
