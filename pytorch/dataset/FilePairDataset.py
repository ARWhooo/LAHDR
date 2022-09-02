from torch.utils import data
import utils.image_io as iio
from .argumentation import argumentation
from utils.misc import list_file_pairs


class FilePairDataset(data.Dataset):
    def __init__(self, pair_locs, pair_suffices, options):
        self.locs = pair_locs
        self.suffix = pair_suffices
        self.options = options
        assert len(pair_locs) == 2, 'FilePairDataset only supports pairs with 2 entries.'
        assert len(pair_locs) == len(pair_suffices), 'Incompatible locs and suffices list.'
        self.list1, self.list2, self.count = list_file_pairs(self.locs[0], self.locs[1], self.suffix[0], self.suffix[1])

    def __getitem__(self, item):
        f1 = iio.load(self.list1[item])
        f2 = iio.load(self.list2[item])
        contents = argumentation([f1, f2], self.options)
        return contents

    def __len__(self):
        return self.count
