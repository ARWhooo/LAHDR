from torch.utils import data
import h5py
from .argumentation import argumentation
import numpy as np
from .H5FetchDataset import H5FetchDataset


class ExposureBiasDataset(H5FetchDataset):
    def __init__(self, dataSrc, fetchKeys, target_idx, options):
        super().__init__(dataSrc, fetchKeys, options)
        fetches_cnt = len(fetchKeys)
        self.fetch_inputs = [fetchKeys[k] for k in range(fetches_cnt) if k != target_idx]
        self.inputs_count = fetches_cnt - 1
        self.fetch_label = fetchKeys[target_idx]

    def __getitem__(self, item):
        result = []
        id = np.random.randint(self.inputs_count)
        tmp = np.array(self.dataset[self.fetch_inputs[id]][item], dtype=np.float32)
        lbl = np.array(self.dataset[self.fetch_label][item], dtype=np.float32)
        result.append(lbl)
        result.append(tmp)
        result = argumentation(result, self.options)
        return result
