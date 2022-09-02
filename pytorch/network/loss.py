import torch
from torchvision import models
import torch.nn.functional as F
import utils.algorithm as alg


class TVRegulation(torch.nn.Module):
    def __init__(self, norm='l2'):
        super().__init__()
        self.norm = norm

    # y is set for dummy position
    def forward(self, x, y=None):
        if self.norm == 'l2':
            dh = F.pad(torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2), (0, 1, 0, 0), 'reflect')
            dw = F.pad(torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2), (0, 0, 0, 1), 'reflect')
            return torch.mean(torch.pow(dh + dw, 0.5))
        else:
            dh = F.pad(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]), (0, 1, 0, 0), 'reflect')
            dw = F.pad(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]), (0, 0, 0, 1), 'reflect')
            return torch.mean(0.5 * (dh + dw))


class TVLoss(torch.nn.Module):
    def __init__(self, norm='l2'):
        super().__init__()
        self.norm = norm

    def forward(self, x, y):
        dh_x = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0), 'reflect')
        dw_x = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1), 'reflect')
        dh_y = F.pad(y[:, :, :, 1:] - y[:, :, :, :-1], (0, 1, 0, 0), 'reflect')
        dw_y = F.pad(y[:, :, 1:, :] - y[:, :, :-1, :], (0, 0, 0, 1), 'reflect')

        if self.norm == 'l2':
            dh = torch.pow(dh_x - dh_y, 2)
            dw = torch.pow(dw_x - dw_y, 2)
            return torch.mean(torch.pow(dh + dw, 0.5))
        else:
            dh = torch.abs(dh_x - dh_y)
            dw = torch.abs(dw_x - dw_y)
            return torch.mean(0.5 * (dh + dw))


class ContentLoss(torch.nn.Module):
    def __init__(self, norm='l2'):
        super().__init__()
        self.norm = norm
        if norm == 'l2':
            self.c = torch.nn.MSELoss()
        else:
            self.c = torch.nn.L1Loss()

    def forward(self, x, y):
        return self.c(x, y)


class ColorLoss(torch.nn.Module):
    def forward(self, x, y):
        # x_ = torch.norm(x, p=2, dim=1, keepdim=True)
        # y_ = torch.norm(y, p=2, dim=1, keepdim=True)
        # x = x / x_
        # y = y / y_
        return 1.0 - torch.mean(torch.cosine_similarity(x, y, dim=1))


class VGGNet(torch.nn.Module):
    def __init__(self, layers=None):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.select = ['0', '5', '10', '19', '28']
        if layers is not None:
            self.select = [self.select[i] for i in layers]
        self.vgg = models.vgg19(pretrained=True).features

    def forward(self, x):
        """Extract 5 conv activation maps from an input image.

        Args:
            x: 4D tensor of shape (1, 3, height, width).

        Returns:
            features: a list containing 5 conv activation maps.
        """
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features


class VGGLoss(torch.nn.Module):
    def __init__(self, layers=None, weights=None, norm='l2'):
        super(VGGLoss, self).__init__()
        self.vgg = VGGNet(layers)
        if weights is None:
            self.weights = [1.0 / len(self.vgg.select)] * len(self.vgg.select)
        else:
            self.weights = weights
        self.norm = norm

    def forward(self, x, y):
        feat_x = self.vgg(x)
        feat_y = self.vgg(y)
        loss = 0
        if self.norm == 'l2':
            func = F.mse_loss
        else:
            func = F.l1_loss
        for i, (x_f, y_f) in enumerate(zip(feat_x, feat_y)):
            loss += self.weights[i] * func(x_f, y_f)
        return loss


class SSIMLoss(torch.nn.Module):
    def __init__(self, ksize=11, sigma=1.5, maxval=1.0):
        super(SSIMLoss, self).__init__()
        self.kernel_size = ksize
        self.sigma = sigma
        self.maximum = maxval

    def forward(self, x, y):
        return 1.0 - alg.ssim(x, y, self.kernel_size, self.sigma, self.maximum)


class MSSIMLoss(torch.nn.Module):
    def __init__(self, ksize=11, sigma=1.5, maxval=1.0):
        super(MSSIMLoss, self).__init__()
        self.kernel_size = ksize
        self.sigma = sigma
        self.maximum = maxval

    def forward(self, x, y):
        return 1.0 - alg.mssim_torch(x, y, self.kernel_size, self.sigma, self.maximum)


class CharbonnierLoss(torch.nn.Module):
    """Charbonnier loss."""
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = torch.add(x, -y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss
