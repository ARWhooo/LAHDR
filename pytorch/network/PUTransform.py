import torch as pt
import torch.nn as nn


# Copied from: https://github.com/gfxdisp/pu_pieapp
class PUTransform(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.P_min = 0.3176
        self.P_max = 1270.1545
        self.logL_min = -1.7647
        self.logL_max = 8.9979
        self.sixth_order = pt.tensor(
            [-2.76770417e-03, 5.45782779e-02, -1.31306184e-01, -4.05827702e+00, 3.74004810e+01, 2.84345651e+01,
             5.15162034e+01])
        self.third_order = pt.tensor([2.5577829, 17.73608751, 48.96952155, 45.55950728])
        self.epsilon = 1e-8

    def forward(self, im, im_type='sdr', lum_top=100, lum_bottom=0.5):

        im = self.apply_disp_model(im, im_type, lum_top=lum_top, lum_bottom=lum_bottom)
        im = self.clamp_image(im)
        im = self.apply_pu(im)
        im = self.scale(im)
        return im

    def apply_disp_model(self, im, im_type, lum_top=100.0, lum_bottom=0.5):
        if im_type == 'hdr':
            return im
        else:
            return (lum_top - lum_bottom) * ((im / 255.0) ** 2.2) + lum_bottom

    def clamp_image(self, im):
        epsilon = 1e-8
        return pt.clamp(pt.log10(pt.clamp(im, epsilon, None)), self.logL_min, self.logL_max)

    def apply_pu(self, img):
        third_ord = self.third_order[0] * img ** 3 + self.third_order[1] * img ** 2 + self.third_order[2] * img + \
                    self.third_order[3]
        sixth_ord = self.sixth_order[0] * img ** 6 + self.sixth_order[1] * img ** 5 + self.sixth_order[2] * img ** 4 + \
                    self.sixth_order[3] * img ** 3 + self.sixth_order[4] * img ** 2 + self.sixth_order[5] * img + \
                    self.sixth_order[6]

        return (img >= 0.8).int() * sixth_ord + (img < 0.8).int() * third_ord

    def scale(self, x):
        """
        scale x to values between 0 and 1
        """
        return (x - self.P_min) / (self.P_max - self.P_min)
