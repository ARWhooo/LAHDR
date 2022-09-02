DEFAULT_TRAIN_ARGUMENTATIONS = {
        'resize': 128,
        'min_allowed_value': 1e-6,
        'max_allowed_value': 10000,
        'need_resize': False,
        'random_crop': False,
        'need_normalize': False,
        'explicit_clipping': False,
        'flipud': True,
        'fliplr': True,
        'rotate': False
}
DEFAULT_TEST_ARGUMENTATIONS = {
        'resize': 128,
        'min_allowed_value': 1e-6,
        'max_allowed_value': 10000,
        'need_resize': False,
        'random_crop': False,
        'need_normalize': False,
        'explicit_clipping': False,
        'flipud': False,
        'fliplr': False,
        'rotate': False
}


class DefaultConfig(object):
    def __init__(self):=
        self.model_name = 'ResBlock'
        self.model = 'ResBlock' 
        self.model_parameters = {
            'in_ch': 3,
            'out_ch': 3,
            'lat_ch': 8,
            'ksize': 3,
            'use_bias': False,
            'layers': 5,
            'act': 'relu',
            'norm': 'none'
        }
        self.adaptive_resume = True

        # Loss and evaluation design
        self.loss = 'my_loss'
        self.loss_parameters = {'c_weight': 1.0, 'r_weight': 1.0, 't_weight': 1.0}
        self.metrics = ['D-PQPSNR', 'D-VDP2']
        self.metric_mu_val = 5000.0
        self.vdp2_matlab_path = '../hdrvdp-2.2.1'

        # Dataloader
        self.checkpoint_dir = './checkpoints'
        self.save_location = './results'
        self.dataset = {
            'train': {
                'dataset_type': 'H5FetchDataset',
                'data_source': '/data/My_Train_dataset.h5',
                'fetch_keys': ['label', 'sample'],
                'argumentations': DEFAULT_TRAIN_ARGUMENTATIONS
            },
            'test': {
                'dataset_type': 'H5FetchDataset',
                'data_source': '/data/My_Test_dataset.h5',
                'fetch_keys': ['label', 'sample'],
                'argumentations': DEFAULT_TEST_ARGUMENTATIONS
            }
        }
        # self.dataset_type = 'H5KVDataset'
        # self.data_source = '/data/My_Train_dataset.h5'
        # self.sample_location = '/data/sample'
        # self.label_location = '/data/label'
        # self.sample_suffices = ".*.(jpg|png|jpeg|bmp)"
        # self.label_suffices = ".*.(pfm|exr|hdr|dng)"
        # self.fetch_keys = ['LDR', 'HDR']
        # self.exclude_keys = []

        # Train and test settings
        self.batch_size = 16
        self.shuffle = True
        self.num_workers = 0
        self.epochs = 120
        self.save_interval = 20
        self.test_interval = 20
        self.gpu_ids = '0'
        self.print_iter_result = False
        self.save_handle = 'save_results'
        self.procedure_handle = 'procedure'

        # Optimizer and scheduler settings
        self.optimizer = 'adam'
        self.decay_type = 'step'  # scheduler
        self.learning_rate = 1e-4  # initial learning rate
        self.opt_momentum = 0.0
        self.opt_beta1 = 0.9
        self.opt_beta2 = 0.999
        self.opt_epsilon = 1e-8
        self.sch_decay_gamma = 0.1
        self.sch_decay_step = 10  # [5, 10, 15, 20, 25, 40, 60, 80, 100, 120]

    def load(self, kwargs, verbose=True):
        for k in kwargs.keys():
            if verbose:
                if not hasattr(self, k):
                    print("Additional attribute added to the options: %s" % k)
            setattr(self, k, kwargs[k])

    def parse(self, kwargs=None, verbose=True):
        options = {}
        if isinstance(kwargs, dict):
            self.load(kwargs, verbose)

        if verbose:
            print('User configs:')
        for k in self.__dict__.keys():
            if not k.startswith('__'):
                if verbose:
                    print_config(k, getattr(self, k), '')
                    # print(k + ': ', getattr(self, k))
                options[k] = getattr(self, k)

        return options


def print_config(name, item, prefix=''):
    if isinstance(item, dict):
        print(prefix + name + ':')
        for n in item.keys():
            print_config(n, item[n], prefix + '\t')
    else:
        print(prefix + name + ': ' + str(item))
