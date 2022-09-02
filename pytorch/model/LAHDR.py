import torch
import torch.nn as nn
from .EnhanceNet import EnhanceNet
from .EBNet import EBNet, GlobalLightRectification
from .DeNet import DeNet
from .FuseNet import FuseNet, get_multi_images
from network import load_checkpoint


class LAHDR(nn.Module):
    def __init__(self, need_denoise=True, trained_configs=None):
        super().__init__()
        self.gamma1 = 1.5
        self.gamma2 = 1.5
        self.mu = 200.0
        self.eb_factor = 3.0
        self.norm_enable = False
        if need_denoise:
            self.candidate_denoise = DeNet(3, 3, int_layers=32, layers=5, ksize=3)
        else:
            self.candidate_denoise = nn.Identity()
        self.denoise = need_denoise
        self.enhancenet = EnhanceNet(3, gf_kernel=5, int_layers=32, layers=4, ksize=3)
        self.ebnet = EBNet(int_layers=32, hist_layers=64)
        self.fusenet = FuseNet(3, in_ch=3, base_ch=16, depth=3, seperate=False, mode='concat', bias=False,
                               act='prelu')
        if trained_configs is not None:
            self._init_sub_networks(trained_configs)
        self.eboffset = GlobalLightRectification(mu=self.mu, factor=self.eb_factor, norm_enable=self.norm_enable)

    def _init_sub_networks(self, configs):
        self.gamma1 = configs['gamma1']
        self.gamma2 = configs['gamma2']
        self.mu = configs['mu']
        self.eb_factor = configs['bias_factor']
        self.norm_enable = configs['bias_norm_enable']
        load_checkpoint(self.enhancenet, configs['EnhanceNet'])
        load_checkpoint(self.ebnet, configs['EBNet'])
        load_checkpoint(self.fusenet, configs['FuseNet'])
        if self.denoise:
            load_checkpoint(self.candidate_denoise, configs['DeNet'])

    def forward(self, x):
        enh = self.enhancenet(x)
        den = torch.clamp(self.candidate_denoise(enh), 0, 1)
        eb = self.ebnet(x)
        ret, eb = self.eboffset(den, eb)
        eb = eb.squeeze(-1).squeeze(-1)
        r0, rplus, rminus = get_multi_images(ret, self.gamma1, self.gamma2)
        return self.fusenet([r0, rplus, rminus]), eb, r0, rplus, rminus

    def train(self, mode=True):
        if mode:
            print('Warning: directly training the complete LA-HDR framework is not suggested. Consider '
                  'training the sub-networks first, and then load the trained models by trained_configs.')
        return super().train(mode)
