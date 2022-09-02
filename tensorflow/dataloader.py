import abc
import os, re
import numpy as np
import random
import h5py
import image_io as iio
import math
import utils as utl


class DataLoader(metaclass=abc.ABCMeta):

    def __init__(self, params, batchsize, epochs, map_func, shuffle=True):
        self.iter_content = None
        self.batchsize = batchsize
        self.total_size = self._initloader(params)
        self.epochs = epochs
        self.cur_order = list(range(self.total_size))
        self.cur_idx = 0
        self.cur_batch_idx = 0
        self.map_func = map_func
        self.shuffle = shuffle
        self.get_random_order()
        '''
        if self.total_size <= self.batchsize:
            self.batchsize = self.total_size
            print("Dataset volume is smaller than the batch size. Batch size is now set to %d." % self.total_size)
        '''

    def get_totalsize(self):
        return self.total_size

    def get_random_order(self):
        if self.shuffle:
            random.shuffle(self.cur_order)

    def iter_batch(self):
        indices = self._pre_process()
        self.iter_content = self._process(indices)
        self._post_process()
        return self.map_func(self.iter_content)

    @abc.abstractmethod
    def _initloader(self, params):
        return self.batchsize

    @abc.abstractmethod
    def _process(self, indices):
        pass

    def _pre_process(self):
        if self.cur_batch_idx + self.batchsize > self.total_size:
            end = self.total_size
            remains = self.batchsize + self.cur_batch_idx - self.total_size
            indices = list(range(self.cur_batch_idx, end))
            for j in range(remains):
                indices.append(random.randint(0, self.total_size - 1))
        else:
            end = self.cur_batch_idx + self.batchsize
            indices = list(range(self.cur_batch_idx, end))
        return indices

    def _post_process(self):
        self.cur_batch_idx += self.batchsize
        if self.cur_batch_idx >= self.total_size:
            self.cur_idx += 1
            self.cur_batch_idx = 0
            self.get_random_order()

    def is_eof(self):
        return (self.cur_idx + 1) > self.epochs

    def get_cur_epoch(self):
        return self.cur_idx

    def get_cur_batchnum(self):
        return np.ceil(float(self.cur_batch_idx) / float(self.batchsize)).astype(np.uint16)

    def get_total_batchnum(self):
        return int(np.ceil(float(self.total_size) / float(self.batchsize)))

    def get_batchsize(self):
        return self.batchsize

    def get_total_epochs(self):
        return self.epochs


class PairedDataFromDir(DataLoader):
    def __init__(self, ldr_dir, hdr_dir, batchsize, epochs, map_func, shuffle=True):
        super(PairedDataFromDir, self).__init__([ldr_dir, hdr_dir], batchsize, epochs, map_func, shuffle)

    def _initloader(self, params):
        ldr_dir = params[0]
        hdr_dir = params[1]
        t1 = os.listdir(ldr_dir)
        t2 = os.listdir(hdr_dir)
        reg_ldr = re.compile(".*.(jpg|png|jpeg|bmp)")
        reg_hdr = re.compile(".*.(pfm|exr|hdr|dng)")
        ldr_list = [f.split('.') for f in t1 if reg_ldr.match(f)]
        hdr_list = [f.split('.') for f in t2 if reg_hdr.match(f)]
        l_list = []
        h_list = []
        ldr_list.sort(key=(lambda x: x[0]))
        hdr_list.sort(key=(lambda x: x[0]))
        if len(ldr_list) != len(hdr_list):
            print("Error: dataset pairs count uneven.")
            exit(1)
        for i in range(len(ldr_list)):
            if ldr_list[i][0] != hdr_list[i][0]:
                print("Error: dataset picture names unpair.")
                exit(1)
            l_list.append(ldr_dir + '/' + ldr_list[i][0] + '.' + ldr_list[i][1])
            h_list.append(hdr_dir + '/' + hdr_list[i][0] + '.' + hdr_list[i][1])

        self.inp_sample = l_list
        self.out_sample = h_list
        return len(ldr_list)

    def _process(self, indices):
        ldr = []
        hdr = []
        for i in indices:
            ldr.append(self.inp_sample[self.cur_order[i]])
            hdr.append(self.out_sample[self.cur_order[i]])

        return [ldr, hdr]


class HDRDataLoader(DataLoader):
    def __init__(self, dataSrc, batchsize, epochs, map_func, shuffle=True):
        super(HDRDataLoader, self).__init__(dataSrc, batchsize, epochs, map_func, shuffle)

    def _initloader(self, params):
        if type(params) is list:
            whole_lost = []
            for loc in params:
                whole_lost.append(self.get_full_filelist(loc))
            self.hdr_samples = whole_lost
            self.hdr_samples.sort()
            return len(self.hdr_samples)
        else:
            self.hdr_samples = self.get_full_filelist(params)
            self.hdr_samples.sort()
            return len(self.hdr_samples)

    def _process(self, indices):
        hdr = []
        for i in indices:
            hdr.append(self.hdr_samples[self.cur_order[i]])

        return hdr

    def get_full_filelist(self, hdr_dir):
        t2 = os.listdir(hdr_dir)
        reg_hdr = re.compile(".*.(pfm|exr|hdr|dng)")
        hdr_list = [f for f in t2 if reg_hdr.match(f)]
        h_list = []
        for names in hdr_list:
            h_list.append(os.path.join(hdr_dir, names))
        return h_list


class SingleDataFromH5(DataLoader):
    def __init__(self, dataSrc, batchsize, epochs, fetches, map_func, shuffle=True):
        self.fetches = fetches
        super(SingleDataFromH5, self).__init__(dataSrc, batchsize, epochs, map_func, shuffle)

    def _initloader(self, params):
        dataSrc = params
        self.data_source = dataSrc
        try:
            self.dataset = h5py.File(dataSrc, 'r')
        except OSError:
            print('Invalid dataset path.')
            exit(-1)
        count = self.dataset[self.fetches[0]].shape[0]
        return count

    def _process(self, indices):
        hdr = {}
        for it in self.fetches:
            it_contents = []
            for i in indices:
                it_contents.append(self.dataset[it][self.cur_order[i]])
            hdr[it] = it_contents
        return hdr


class KVDataFromH5(DataLoader):
    def __init__(self, dataSrc, batchsize, epochs, fetches, map_func, shuffle=True):
        self.fetches = fetches
        super(KVDataFromH5, self).__init__(dataSrc, batchsize, epochs, map_func, shuffle)

    def _initloader(self, params):
        dataSrc = params
        self.data_source = dataSrc
        try:
            self.dataset = h5py.File(dataSrc, 'r')
        except OSError:
            print('Invalid dataset path.')
            exit(-1)
        count = len(self.dataset[self.fetches[0]])
        self.key_list = list(self.dataset[self.fetches[0]].keys())
        return count

    def _process(self, indices):
        hdr = {}
        for it in self.fetches:
            handle = self.dataset[it]
            it_contents = []
            for i in indices:
                content = np.array(handle[self.key_list[self.cur_order[i]]])
                it_contents.append(content)
            hdr[it] = it_contents
        return hdr


class SelectedKVLoader(KVDataFromH5):
    def __init__(self, dataSrc, batchsize, epochs, fetches, map_func, exclude_src, log_loc, shuffle=True):
        self.exc = exclude_src
        self.log = log_loc
        super(SelectedKVLoader, self).__init__(dataSrc, batchsize, epochs, fetches, map_func, shuffle)

    def _initloader(self, params):
        dataSrc = params
        self.data_source = dataSrc
        try:
            self.dataset = h5py.File(dataSrc, 'r')
        except OSError:
            print('Invalid dataset path.')
            exit(-1)
        key_list = list(self.dataset[self.fetches[0]].keys())
        sz = len(key_list)
        for exc in self.exc:
            key_list = [it for it in key_list if exc not in it]
            sz_ = len(key_list)
            utl.save_to_file(self.log, 'Excluding sample category: %s, %d samples removed.' % (exc, sz - sz_))
            sz = sz_
        self.key_list = key_list
        utl.save_to_file(self.log, 'Altogether %s samples remained for training.' % sz)
        return len(self.key_list)


def argument(contents, args):
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
        if args['need_resize']:
            if args['random_crop']:
                ldr = ldr[hs:(hs + resize), ws:(ws + resize), :]
            else:
                ldr = iio.image_resize(ldr, [resize, resize])
        if args['need_normalize']:
            if args['explicit_clipping']:
                ldr = ldr.astype(np.float32)
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


def map_H5(cont_dict, args=None, proc=argument):
    if args is None:
        args = default_argumentation_args()

    fetches = list(cont_dict.keys())
    batches = len(cont_dict[fetches[0]])
    tmp = []
    for i in range(batches):
        for it in fetches:
            content = cont_dict[it][i]
            tmp.append(content)
        tmp = proc(tmp, args)
        for it in range(len(fetches)):
            key = fetches[it]
            cont_dict[key][i] = tmp[it]
        tmp.clear()

    out = {}
    for it in fetches:
        out[it] = np.stack(cont_dict[it], axis=0)
    return out
