from torch.utils import data
import h5py
from .argumentation import argumentation
import numpy as np


class H5FetchDataset(data.Dataset):
    def __init__(self, dataSrc, fetchKeys, options):
        self.data_source = dataSrc
        try:
            self.dataset = h5py.File(dataSrc, 'r')
        except OSError:
            print('Invalid dataset path.')
            exit(-1)
        self.count = self.dataset[fetchKeys[0]].shape[0]
        self.fetches = fetchKeys
        self.options = options

    def __getitem__(self, item):
        result = []
        for f in self.fetches:
            tmp = np.array(self.dataset[f][item], dtype=np.float32)
            result.append(tmp)
        result = argumentation(result, self.options)
        return result

    def __len__(self):
        return self.count
