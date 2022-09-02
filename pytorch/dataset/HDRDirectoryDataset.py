import os
from torch.utils import data
import cv2
from utils.misc import process_path
from utils.algorithm import map_range, random_tonemap
import numpy as np
from .argumentation import argumentation


class HDRDirectoryDataset(data.Dataset):
    def __init__(
        self,
        hdr_loc,
        options,
        data_extensions=('.hdr', '.exr'),
        preprocess=None,
    ):
        super(HDRDirectoryDataset, self).__init__()
        self.options = options
        data_root_path = process_path(hdr_loc)
        self.file_list = []
        for root, _, fnames in sorted(os.walk(data_root_path)):
            for fname in fnames:
                if any(
                    fname.lower().endswith(extension)
                    for extension in data_extensions
                ):
                    self.file_list.append(os.path.join(root, fname))
        if len(self.file_list) == 0:
            msg = 'Could not find any files with extensions:\n[{0}]\nin\n{1}'
            raise RuntimeError(
                msg.format(', '.join(data_extensions), data_root_path)
            )

        if preprocess is None:
            self.preprocess = default_preprocess_func
        else:
            self.preprocess = preprocess

    def __getitem__(self, index):
        dpoint = cv2.imread(
            self.file_list[index], flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR
        )
        dpoint = np.flip(dpoint, 2)
        if self.preprocess is not None:
            dpoint = self.preprocess(dpoint, self.options)
        return dpoint

    def __len__(self):
        return len(self.file_list)


def default_preprocess_func(hdr, opt):
    hdr = cv2.resize(hdr, (opt['resize'], opt['resize']))
    hdr = map_range(hdr)
    ldr = random_tonemap(hdr)
    return argumentation([ldr, hdr], opt)
