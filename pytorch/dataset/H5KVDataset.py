from torch.utils import data
import h5py
from .argumentation import argumentation
import numpy as np


class H5KVDataset(data.Dataset):
    def __init__(self, dataSrc, fetchKV, options, excludeKeys=None):
        self.data_source = dataSrc
        try:
            self.dataset = h5py.File(dataSrc, 'r')
        except OSError:
            print('Invalid dataset path.')
            exit(-1)
        self.fetches = fetchKV
        key_list = list(self.dataset[self.fetches[0]].keys())
        self.exc = excludeKeys if excludeKeys is not None else []
        sz = len(key_list)
        for exc in self.exc:
            key_list = [it for it in key_list if exc not in it]
            sz_ = len(key_list)
            print('Excluding sample category: %s, %d samples removed.' % (exc, sz - sz_))
            sz = sz_
        self.key_list = key_list
        self.count = len(self.key_list)
        self.options = options

    def __len__(self):
        return self.count

    def __getitem__(self, item):
        result = []
        for f in self.fetches:
            tmp = np.array(self.dataset[f][self.key_list[item]], dtype=np.float32)
            result.append(tmp)
        result = argumentation(result, self.options)
        return result
