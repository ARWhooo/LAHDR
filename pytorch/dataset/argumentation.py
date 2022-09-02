import utils.image_io as iio
import math, random
import numpy as np
import torch


def argumentation(contents, args):
    # Data Argumentation
    resize = args['resize']
    sample = contents[0]
    h, w, _ = sample.shape
    hs = math.ceil(random.uniform(0, h - resize))
    ws = math.ceil(random.uniform(0, w - resize))
    flipud = random.uniform(0, 1) < 0.5
    fliplr = random.uniform(0, 1) < 0.5
    seed = random.uniform(0, 1)
    if 0.25 < seed <= 0.5:  # 90 Degree
        angle = 90
    elif 0.5 < seed <= 0.75:
        angle = 180
    elif 0.75 < seed <= 1:
        angle = 270
    else:
        angle = 0
    minval = args['min_allowed_value']
    maxval = args['max_allowed_value']
    for i in range(len(contents)):
        ldr = contents[i]
        if 'str' in str(type(ldr)) or len(ldr.shape) != 3:
            if not 'str' in str(type(ldr)):
                contents[i] = torch.Tensor(contents[i])
            continue
        ldr = ldr.astype(np.float32)
        if args['need_resize']:
            if args['random_crop']:
                ldr = ldr[hs:(hs + resize), ws:(ws + resize), :]
            else:
                ldr = iio.image_resize(ldr, [resize, resize])
        if args['need_normalize']:
            if args['explicit_clipping']:
                ldr[ldr > maxval] = maxval
                ldr[ldr < minval] = minval
                ldr = ldr / maxval
            else:
                ldr = iio.map_range(ldr, minval, maxval)
        if args['flipud']:
            if flipud:
                ldr = iio.flip_ud(ldr)
        if args['fliplr']:
            if fliplr:
                ldr = iio.flip_lr(ldr)
        if args['rotate']:
            ldr = iio.image_rotate(ldr, angle)
        ldr = torch.from_numpy(ldr.transpose((2, 0, 1))).contiguous()
        contents[i] = ldr
    return contents


def default_argumentation_args():
    args = {'need_normalize': False,
            'explicit_clipping': True,
            'min_allowed_value': 1e-7,
            'max_allowed_value': 10000.0,
            'need_resize': True,
            'resize': 256,
            'random_crop': True,
            'flipud': True,
            'fliplr': True,
            'rotate': True}
    return args
