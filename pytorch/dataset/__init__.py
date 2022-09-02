from torch.utils.data.dataloader import DataLoader
from .H5FetchDataset import H5FetchDataset
from .H5KVDataset import H5KVDataset
from .FilePairDataset import FilePairDataset
from .HDRDirectoryDataset import HDRDirectoryDataset
from .ExposureBiasDataset import ExposureBiasDataset
from .argumentation import default_argumentation_args, argumentation


def get_dataset(opt_, stage='train'):
    opt = opt_['dataset'][stage]
    datatype = opt['dataset_type']
    argument = opt['argumentations']
    if datatype == 'FilePairDataset':
        locs = [opt['sample_location'], opt['label_location']]
        suffices = [opt['sample_suffices'], opt['label_suffices']]
        dataset = FilePairDataset(locs, suffices, argument)
    elif datatype == 'H5KVDataset':
        src = opt['data_source']
        fetch = opt['fetch_keys']
        exc = opt['exclude_keys']
        dataset = H5KVDataset(src, fetch, argument, exc)
    elif datatype == 'H5FetchDataset':
        src = opt['data_source']
        fetch = opt['fetch_keys']
        dataset = H5FetchDataset(src, fetch, argument)
    elif datatype == 'HDRDirectoryDataset':
        src = opt['label_location']
        dataset = HDRDirectoryDataset(src, argument)
    elif datatype == 'ExposureBiasDataset':
        src = opt['data_source']
        fetch = opt['fetch_keys']
        idx = opt['target_idx']
        dataset = ExposureBiasDataset(src, fetch, idx, argument)
    else:
        raise Exception('Invalid dataset type: %s. Check the available dataset types in the data folder.' % datatype)

    if stage == 'train':
        batches = opt_['batch_size']
        shuffle = opt_['shuffle']
        workers = opt_['num_workers']
    else:
        batches = 1
        shuffle = False
        workers = 0

    return DataLoader(dataset, batches, shuffle, num_workers=workers)
