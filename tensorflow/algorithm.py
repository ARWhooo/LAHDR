import tensorflow as tf
import numpy as np
import os, cv2
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import compare_ssim
import random


PU_PLUT_FILENAME = './data/PU_P_lut.npy'
PU_LLUT_FILENAME = './data/PU_l_lut.npy'
COMP_MAXVAL = 1e4
COMP_MINVAL = 1e-6
crf_txt_file = './data/dorfCurves.txt'


class pu_luts:
    def __init__(self):
        self.init_flag = False

    def init(self, p_lut_loc=None, l_lut_loc=None):
        if p_lut_loc is None:
            p_lut_loc = PU_PLUT_FILENAME
        if l_lut_loc is None:
            l_lut_loc = PU_LLUT_FILENAME
        self.P_lut = np.load(p_lut_loc)
        self.l_lut = np.load(l_lut_loc)
        self.init_flag = True

    def get_P_lut(self):
        if not self.init_flag:
            self.init()
        return self.P_lut

    def get_l_lut(self):
        if not self.init_flag:
            self.init()
        return self.l_lut


PU_lut = pu_luts()
SUGGEST_DORF_SEGMENTS ={
                        # 'A': (0, 52),
                        'B': (52, 72),
                        # 'C': (72, 112),
                        'D': (112, 142),
                        # 'E': (142, 152),
                        'F': (152, 172),
                        'G': (172, 200)
}  # Type 1: A, C, E with bigger contrast, Type 2: B, D, F, G with lower contrast


class CRFIdxGenerator:
    def __init__(self):
        self.ent = list(SUGGEST_DORF_SEGMENTS.keys())
        random.shuffle(self.ent)
        self.idx = -1

    def get(self):
        self.idx += 1
        if self.idx > (len(SUGGEST_DORF_SEGMENTS) - 1):
            self.idx = 0
            random.shuffle(self.ent)
        rang = SUGGEST_DORF_SEGMENTS[self.ent[self.idx]]
        return np.random.randint(rang[0], rang[1])


crf_idx_generator = CRFIdxGenerator()


class DoRF_CRF:
    def __init__(self, dorf_crf_file):
        self.init_flag = False
        self.crf_file = dorf_crf_file

    def parse_dorf(self, dorf_txt_loc):
        with open(os.path.join(dorf_txt_loc), 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]

        i = [lines[idx + 3] for idx in range(0, len(lines), 6)]
        b = [lines[idx + 5] for idx in range(0, len(lines), 6)]
        type = [lines[idx] for idx in range(0, len(lines), 6)]
        info = [lines[idx + 1] for idx in range(0, len(lines), 6)]

        i = [ele.split() for ele in i]
        b = [ele.split() for ele in b]

        i = np.float32(i)
        b = np.float32(b)
        self.init_flag = True

        return i, b, type, info

    def random_crf(self, norm_inp, verbose=True):
        if not self.init_flag:
            self.ind, self.crf, self.type, _ = self.parse_dorf(self.crf_file)
        choice = np.random.randint(0, self.crf.shape[0])
        if verbose:
            print('Chosen CRF curve: %s.' % self.type[choice])
        return np.interp(norm_inp, self.ind[choice, :], self.crf[choice, :])

    def do_crf(self, norm_inp, id, verbose=False):
        if not self.init_flag:
            self.ind, self.crf, self.type, _ = self.parse_dorf(self.crf_file)
        if id >= 201:
            raise ValueError('Only 201 CRFs included in DoRF database, but given id = %d.' % id)
        choice = id
        if verbose:
            print('Chosen CRF curve: %s.' % self.type[choice])
        return np.interp(norm_inp, self.ind[choice, :], self.crf[choice, :])

    def get_crf_curve(self, idx):
        if not self.init_flag:
            self.ind, self.crf, self.type, _ = self.parse_dorf(self.crf_file)
        if idx >= 201:
            raise ValueError('Only 201 CRFs included in DoRF database, but given id = %d.' % id)
        return self.type[idx], self.ind[idx, :], self.crf[idx, :]


camera_transfer = DoRF_CRF(crf_txt_file)


def mu_law_inverse(x, mu=1200.0):
    mu = float(mu)
    if 'tensorflow' in str(type(x)):
        return tf.log(1 + mu * x) / tf.log(1 + mu)
    else:
        return np.log(1 + mu * x) / np.log(1 + mu)


def mu_law_forward(x, mu=1200.0):
    mu = float(mu)
    if 'tensorflow' in str(type(x)):
        out = tf.exp(tf.log(1 + mu) * x) - 1
    else:
        out = np.exp(np.log(1 + mu) * x) - 1
    return out / mu


def pu(x):
    l_min = -5.0
    l_max = 10.0
    l = np.log10(np.maximum(np.minimum(np.power(10, l_max), x), np.power(10, l_min)))
    pu_l = 31.9270
    pu_h = 149.9244
    return (np.interp(l, PU_lut.get_l_lut(), PU_lut.get_P_lut()) - pu_l) / (pu_h - pu_l)


def psnr(pre, ref, maxval):
    if 'tensorflow' in str(type(pre)):
        return tf.image.psnr(pre, ref, maxval)
    else:
        MSE = np.mean((pre - ref) ** 2)
        ans = 20 * np.log(maxval) / np.log(10.0) - 10.0 / np.log(10.0) * np.log(MSE)
        return ans


def ssim(pre, ref, maxval):
    if 'tensorflow' in str(type(pre)):
        return tf.image.ssim(pre, ref, maxval)
    else:
        return compare_ssim(pre, ref, data_range=maxval, gradient=False, multichannel=True,
                            gaussian_weights=True, full=False, use_sample_covariance=False,
                            sigma=1.5)


def pq_inverse(inp):
    # make sure the inp is normalized into 0 ~ 1
    # HDR to LDR
    m1 = 0.1593
    m2 = 78.8438
    c1 = 0.8359
    c2 = 18.8516
    c3 = 18.6875

    if 'tensorflow' in str(type(inp)):
        out = tf.pow((c1 + c2 * (tf.pow(inp, m1))) / (1 + c3 * (tf.pow(inp, m1))), m2)
    else:
        out = np.power((c1 + c2 * (np.power(inp, m1))) / (1 + c3 * (np.power(inp, m1))), m2)
    return out


def pq_forward(inp):
    # make sure the inp is normalized into 0 ~ 1
    # LDR to HDR
    M1 = 0.1593
    M2 = 78.8438
    C1 = 0.8359
    C2 = 18.8516
    C3 = 18.6875

    if 'tensorflow' in str(type(inp)):
        out = tf.pow((tf.maximum((tf.pow(inp, 1 / M2) - C1), 0)) / (C2 - C3 * tf.pow(inp, (1 / M2))), 1 / M1)
    else:
        out = np.power((np.maximum((np.power(inp, 1 / M2) - C1), 0)) / (C2 - C3 * np.power(inp, (1 / M2))), 1 / M1)
    return out


def hlg_inverse(inp):
    # make sure the inp is normalized into 0 ~ 1
    # HDR to LDR
    a = 0.17883277
    b = 1 - 4 * a
    c = 0.5 - a * np.log(4*a)
    inp = np.select([inp > 1/12, inp <= 1/12], [a*np.log(12*inp-b)+c, np.sqrt(np.abs(3*inp))])
    return inp


def hlg_forward(inp):
    # make sure the inp is normalized into 0 ~ 1
    # LDR to HDR
    a = 0.17883277
    b = 1 - 4 * a
    c = 0.5 - a * np.log(4 * a)
    inp = np.select([inp > 1 / 2, inp <= 1 / 2], [(np.exp((inp-c) / a) + b) / 12.0, tf.pow(inp, 2.0) / 3.0])
    return inp


def map_range(x, low=0.0, high=1.0):
    return np.interp(x, [x.min(), x.max()], [low, high]).astype(x.dtype)


def mask_low(a, th):
    if 'tensorflow' in str(type(a)):
        return 0.5 * (1 - tf.tanh(200.0 * (a - th)))
    else:
        return 0.5 * (1 - np.tanh(200.0 * (a - th)))


def mask_high(a, th):
    if 'tensorflow' in str(type(a)):
        return 0.5 * (1 + tf.tanh(200.0 * (a - th)))
    else:
        return 0.5 * (1 + np.tanh(200.0 * (a - th)))


def filter_high(a, th):
    return a * mask_low(a, th)


def filter_low(a, th):
    return a * mask_high(a, th)


def get_luminance(rgbimg):
    mat = [0.212639, 0.715169, 0.072192]
    if 'tensorflow' in str(type(rgbimg)):
        l = mat[0] * rgbimg[:, :, :, 0] + mat[1] * rgbimg[:, :, :, 1] + mat[2] * rgbimg[:, :, :, 2]
        return tf.expand_dims(l, axis=-1)
    else:
        l = mat[0] * rgbimg[:, :, 0] + mat[1] * rgbimg[:, :, 1] + mat[2] * rgbimg[:, :, 2]
        return np.expand_dims(l, axis=-1)


def get_saturation(rgbimg):
    if 'tensorflow' in str(type(rgbimg)):
        maxv = tf.reduce_max(rgbimg, axis=-1, keepdims=True)
        minv = tf.reduce_min(rgbimg, axis=-1, keepdims=True)
    else:
        maxv = np.max(rgbimg, axis=-1, keepdims=True)
        minv = np.min(rgbimg, axis=-1, keepdims=True)
    s = (maxv - minv) / (maxv + 1e-6)
    return s


def get_gradient(img):
    if 'tensorflow' in str(type(img)):
        pad_H = tf.pad(img, [[0, 0], [0, 1], [0, 0], [0, 0]], mode='REFLECT')
        pad_W = tf.pad(img, [[0, 0], [0, 0], [0, 1], [0, 0]], mode='REFLECT')
        tv_H = tf.abs(pad_H[:, :-1, :, :] - pad_H[:, 1:, :, :])
        tv_W = tf.abs(pad_W[:, :, :-1, :] - pad_W[:, :, 1:, :])
    else:
        pad_H = np.pad(img, [[0, 1], [0, 0], [0, 0]], mode='reflect')
        pad_W = np.pad(img, [[0, 0], [0, 1], [0, 0]], mode='reflect')
        tv_H = np.abs(pad_H[:-1, :, :] - pad_H[1:, :, :])
        tv_W = np.abs(pad_W[:, :-1, :] - pad_W[:, 1:, :])
    return (tv_H + tv_W) / 2.0


def get_evLevel(rgbimg, mu=0.5, sigma=0.2):
    dem = 2 * sigma * sigma
    if 'tensorflow' in str(type(rgbimg)):
        img = tf.reduce_max(rgbimg, axis=-1, keepdims=True)
        out = -1.0 * tf.square(img - mu)
        return tf.exp(out)
    else:
        img = np.max(rgbimg, axis=-1, keepdims=True)
        out = -1.0 * np.square(img - mu)
        return np.exp(out)


def sobel_edge_mask(inp):
    mask = tf.image.sobel_edges(inp)[:, :, :, :, 0]
    mask = tf.abs(mask)
    mask = mask_high(mask, 0.1)
    mask = tf.reduce_max(mask, axis=-1, keepdims=True)
    return mask


def bcp(img, k, name='bright_channel_prior'):
    with tf.variable_scope(name):
        img = tf.reduce_max(img, axis=-1, keepdims=True)
        return tf.squeeze(tf.nn.max_pool(img, [1, k, k, 1], [1, 1, 1, 1], 'VALID'), axis=-1)


class StretchContrastLevel(object):
    def __init__(self, threshLow=1/255.0, threshHigh=245/255.0, blackLevel=1/255.0, saturateLevel=245/255.0):
        self.th_low = threshLow
        self.th_high = threshHigh
        self.low = blackLevel
        self.high = saturateLevel

    def process(self, img):
        if img.max() < self.th_high:
            th_max = self.high
        else:
            th_max = img.max()
        if img.min() > self.th_low:
            th_min = self.low
        else:
            th_min = img.min()
        img = map_range(img, th_min, th_max)
        return img

    def __call__(self, img):
        return self.process(img)


class CRFProcess(object):
    def __init__(self, stops=0.0, crf_idx=100, randomize=False):
        if randomize:
            crf_idx = np.random.randint(0, 201)
        self.stops = stops
        self.crf = crf_idx

    def process(self, img):
        img = img * (2 ** self.stops)
        img = np.clip(img, 0, 1)
        img = camera_transfer.do_crf(img, self.crf, verbose=False)
        return img

    def get_crf_curve(self):
        return camera_transfer.get_crf_curve(self.crf)

    def __call__(self, img):
        return self.process(img)


class HardClip(object):
    def __init__(self, low_clip=0.0, high_clip=1.0):
        self.low = low_clip
        self.high = high_clip

    def process(self, img):
        return np.clip(img, self.low, self.high)

    def __call__(self, img):
        return self.process(img)


class OffsetMeanLevel(object):
    def __init__(self, mid_level=0.2, randomize=False):
        if randomize:
            mid_level = np.random.uniform(0.15, 0.51)
        self.level = mid_level

    def process(self, img):
        mean = np.mean(img)
        scale = self.level / mean
        img = img * scale
        return img

    def __call__(self, img):
        return self.process(img)


class PercentileClip(object):
    def __init__(self, low_clip=-1.0, high_clip=98.0, normalize=True, randomize=False):
        if randomize:
            low_clip = np.random.uniform(0.1, 2.0)
            high_clip = np.random.uniform(97.0, 99.0)
        if low_clip < 0.0 or low_clip > 100.0:
            self.enable_low = False
        else:
            self.enable_low = True
        self.low = low_clip
        if high_clip < 0.0 or high_clip > 100.0:
            self.enable_high = False
        else:
            self.enable_high = True
        self.high = high_clip
        self.normalize = normalize
        if self.enable_low is False and self.enable_high is False:
            print('Invalid PercentileClip ranges: %.1f - %.1f' % (low_clip, high_clip))
            exit(-1)

    def process(self, img):
        if not self.enable_low:
            high = np.percentile(img, self.high)
            img = np.clip(img, None, high)
        elif not self.enable_high:
            low = np.percentile(img, self.low)
            img = np.percentile(img, low, None)
        else:
            low, high = np.percentile(img, (self.low, self.high))
            img = np.clip(img, low, high)
        if self.normalize:
            return map_range(img, 0.0, 1.0)
        else:
            return img

    def __call__(self, img):
        return self.process(img)


class GammaInverse(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def process(self, img):
        return img ** (1.0 / self.gamma)

    def __call__(self, img):
        return self.process(img)


class EVOffset(object):
    def __init__(self, ev):
        self.ev = ev

    def process(self, img):
        return img * (2 ** (-1.0 * self.ev))

    def __call__(self, img):
        return self.process(img)


class HSVProcess(object):
    def __init__(self, h_func, s_func, v_func):
        self.h = h_func
        self.s = s_func
        self.v = v_func

    def process(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        if self.h is not None:
            hsv[:, :, 0] = self.h(hsv[:, :, 0])
        if self.s is not None:
            hsv[:, :, 1] = self.s(hsv[:, :, 1])
        if self.v is not None:
            hsv[:, :, 2] = self.v(hsv[:, :, 2])
        im = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return im

    def __call__(self, img):
        return self.process(img)


class Transforms:
    def __init__(self, func_list):
        self.funcs = func_list

    def __call__(self, img):
        for func in self.funcs:
            img = func(img)
        return img


class BaseTMO(object):
    def __init__(self, randomize=True):
        self.op = None
        self.params = {}
        self.random = randomize
        self.name = 'BaseTMO'

    def _rand_init(self):
        pass

    def __call__(self, img):
        if self.random:
            self._rand_init()
        return self.op.process(img)


class Reinhard(BaseTMO):
    def __init__(
        self,
        intensity=1.0,
        light_adapt=1.0,
        color_adapt=1.0,
        gamma=2.2,
        randomize=False,
    ):
        super(Reinhard, self).__init__(randomize=randomize)
        self.op = cv2.createTonemapReinhard(
            gamma=gamma,
            intensity=intensity,
            light_adapt=light_adapt,
            color_adapt=color_adapt,
        )
        self.params['gamma'] = gamma
        self.params['intensity'] = intensity
        self.params['light_adapt'] = light_adapt
        self.params['color_adapt'] = color_adapt
        self.name = 'Reinhard'

    def _rand_init(self):
        gamma = np.random.uniform(2.0, 2.2)
        intensity = np.random.uniform(-1.0, 1.0)
        light_adapt = np.random.uniform(0.8, 1.0)
        color_adapt = np.random.uniform(0.0, 0.2)
        self.params['gamma'] = gamma
        self.params['intensity'] = intensity
        self.params['light_adapt'] = light_adapt
        self.params['color_adapt'] = color_adapt
        self.op = cv2.createTonemapReinhard(
            gamma=gamma,
            intensity=intensity,
            light_adapt=light_adapt,
            color_adapt=color_adapt,
        )


class Mantiuk(BaseTMO):
    def __init__(self, saturation=1.0, scale=1.0, gamma=2.2, randomize=False):
        super(Mantiuk, self).__init__(randomize=randomize)
        self.op = cv2.createTonemapMantiuk(
            saturation=saturation, scale=scale, gamma=gamma
        )
        self.params['saturation'] = saturation
        self.params['scale'] = scale
        self.params['gamma'] = gamma
        self.name = 'Mantiuk'

    def _rand_init(self):
        gamma = np.random.uniform(2.0, 2.2)
        scale = np.random.uniform(0.65, 0.85)
        saturation = np.random.uniform(0.85, 1.0)
        self.op = cv2.createTonemapMantiuk(
            saturation=saturation, scale=scale, gamma=gamma
        )
        self.params['saturation'] = saturation
        self.params['scale'] = scale
        self.params['gamma'] = gamma


class Drago(BaseTMO):
    def __init__(self, saturation=1.0, bias=0.85, gamma=2.2, randomize=False):
        super(Drago, self).__init__(randomize=randomize)
        self.op = cv2.createTonemapDrago(
            saturation=saturation, bias=bias, gamma=gamma
        )
        self.params['saturation'] = saturation
        self.params['bias'] = bias
        self.params['gamma'] = gamma
        self.name = 'Drago'

    def _rand_init(self):
        gamma = np.random.uniform(2.0, 2.2)
        bias = np.random.uniform(0.7, 0.9)
        saturation = np.random.uniform(0.85, 1.0)
        self.op = cv2.createTonemapDrago(
            saturation=saturation, bias=bias, gamma=gamma
        )
        self.params['saturation'] = saturation
        self.params['bias'] = bias
        self.params['gamma'] = gamma


class CRFToneMap(BaseTMO):
    def __init__(self, crf_idx=100, stops=0.0, gamma=2.2, randomize=False):
        super(CRFToneMap, self).__init__(randomize=randomize)
        if randomize:
            crf_idx = np.random.randint(0, 201)
            #gamma = np.random.uniform(2.05, 2.35)
        self.stops = stops
        self.crf = crf_idx
        self.gamma = gamma
        name, ind, val = self.get_crf_curve()
        self.params['EV'] = stops
        self.params['gamma'] = gamma
        self.params['crf'] = name
        self.params['crf_ind'] = ind
        self.params['crf_val'] = val
        self.op = self
        self.name = 'CRF'

    def process(self, img):
        img = img * (2 ** self.stops)
        img = np.clip(img, 0, 1)
        img = camera_transfer.do_crf(img, self.crf, verbose=False)
        return img ** (1.0 / self.gamma)

    def _rand_init(self):
        self.crf = np.random.randint(0, 201)
        self.stops = np.random.uniform(-0.5, 0.5)
        self.gamma = np.random.uniform(1.8, 2.2)
        name, ind, val = self.get_crf_curve()
        self.params['EV'] = self.stops
        self.params['gamma'] = self.gamma
        self.params['crf'] = name
        self.params['crf_ind'] = ind
        self.params['crf_val'] = val

    def get_crf_curve(self):
        return camera_transfer.get_crf_curve(self.crf)


TMO_DICT = {
    'crf': CRFToneMap,
    'reinhard': Reinhard,
    'mantiuk': Mantiuk,
    'drago': Drago,
}


def tonemap(img, tmo_name, **kwargs):
    out, _ = TMO_DICT[tmo_name](**kwargs)(img)
    return np.fix(map_range(out) * 255.0) / 255.0


def random_tonemap(x):
    tmos = list(TMO_DICT.keys())
    choice = np.random.randint(0, len(tmos))
    tmo = TMO_DICT[tmos[choice]](randomize=True)
    out = tmo(x)
    out = map_range(out)
    return np.fix(out * 255.0) / 255.0


class Masia17rTMO:
    def __init__(self, gamma=2.2, flag=True):
        self.gamma = gamma
        self.eps = 1e-6
        self.flag = flag

    def __call__(self, normalized_img):
        img = normalized_img
        img = img.astype(np.float32)
        img = img ** self.gamma
        L = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
        L_H = np.exp(np.log(L + self.eps).mean())
        oL1 = self._rid_outliner(L, 1)
        k1 = (np.log(L_H) - np.log(oL1.min())) / (np.log(oL1.max()) - np.log(oL1.min()))
        p_ov = np.prod(L[L > 254.0 / 255.0].shape) / np.prod(L.shape)
        gamma = 2.4379 + 0.2319 * np.log(L_H) - 1.1228 * k1 + 0.0085 * p_ov
        if self.flag:
            return img ** gamma
        else:
            return img ** (1.0 / gamma)

    def _rid_outliner(self, img, p):
        th_min, th_max = np.percentile(img, (p, 100 - p))
        im = img.copy()
        im[im < th_min] = th_min
        im[im > th_max] = th_max
        return im


def get_patch_complexity(pt):
    pt = np.max(pt, axis=-1)
    gre = greycomatrix(pt.astype(np.uint8), distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    level = greycoprops(gre, 'contrast')
    return level[0, 0]
