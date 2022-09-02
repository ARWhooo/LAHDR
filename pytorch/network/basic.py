import os, math, time
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parallel import DistributedDataParallel
import utils.misc as misc
from torchstat import stat
import abc
from utils import probe_gpu, get_records
import importlib


def flop_status(net, shape=(3, 1024, 1024)):
    return stat(net, shape)


# from github repo: CycleISP
def freeze(my_model):
    for p in my_model.parameters():
        p.requires_grad = False


# from github repo: CycleISP
def unfreeze(my_model):
    for p in my_model.parameters():
        p.requires_grad = True


# from github repo: CycleISP
def is_frozen(my_model):
    x = [p.requires_grad for p in my_model.parameters()]
    return not all(x)


def get_all_model_devices(model):
    devices = []
    for mdl in model.state_dict().values():
        if mdl.device not in devices:
            devices.append(mdl.device)
    return devices


def load_checkpoint(my_model, path):
    load_net = torch.load(path, map_location=get_all_model_devices(my_model)[0])
    if 'state_dict' in load_net.keys():
        load_net = load_net['state_dict']
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    my_model.load_state_dict(load_net_clean)


def save_checkpoint(my_model, file):
    tmp_dict = {}
    for key, param in my_model.state_dict().items():
        if key.startswith('module.'):
            key = key[7:]
        tmp_dict[key] = param.cpu()
    torch.save(tmp_dict, file)


# from github repo: CycleISP
def load_epoch(epochfile):
    checkpoint = torch.load(epochfile)
    epoch = checkpoint["epoch"]
    return epoch


def get_model(options, verbose=True):
    model_dt = options['model']
    params = options['model_parameters']
    md = importlib.import_module('model.' + model_dt)
    model = getattr(md, model_dt)(**params)
    model_loc = options['checkpoint_dir']
    model_name = options['model_name']
    if 'model_initmode' in options.keys():
        initmode = options['model_initmode']
    else:
        initmode = 'xavier'
    mdl = BaseModel(model, model_name, model_loc, initmode, verbose)
    if 'model_initmode' in options.keys():
        mdl.parameters_init()
    return mdl


def get_loss(options):
    loss_dt = options['loss']
    params = options['loss_parameters']
    md = importlib.import_module('loss.' + loss_dt)
    return getattr(md, loss_dt)(**params)


# from github repo: DASR
def make_optimizer(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args['optimizer'] == 'SGD' or args['optimizer'] == 'sgd':
        optimizer_function = torch.optim.SGD
        kwargs = {'lr': args['learning_rate'], 'momentum': args['opt_momentum']}
    elif args['optimizer'] == 'ADAM' or args['optimizer'] == 'adam':
        optimizer_function = torch.optim.Adam
        kwargs = {
            'lr': args['learning_rate'],
            'betas': (args['opt_beta1'], args['opt_beta2']),
            'eps': args['opt_epsilon']
        }
    else:
        optimizer_function = torch.optim.RMSprop
        kwargs = {'lr': args['learning_rate'], 'eps': args['opt_epsilon']}
    return optimizer_function(trainable, **kwargs)


# from github repo: DASR
def make_scheduler(args, my_optimizer):
    if args['decay_type'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            my_optimizer,
            step_size=args['sch_decay_step'],
            gamma=args['sch_decay_gamma'],
        )
    elif args['decay_type'] == 'multi_step':   ## milestones
        milestones = args['sch_decay_step'].split(',')
        #milestones.pop(0)
        milestones = list(map(lambda x: int(x.strip()), milestones))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args['sch_decay_gamma']
        )
    elif args['decay_type'] == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            my_optimizer,
            gamma=args['sch_decay_gamma']
        )
    else:
        scheduler = None
        print('No scheduler assigned for the training process.')
    return scheduler


def model_parameter_init_normal(model_):
    for m in model_.modules():
        if isinstance(m, nn.Conv2d):
            B, C, H, W = m.weight.size()
            m.weight.data.normal_(0, math.sqrt(2. / (C * H * W)))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.05)
            if m.bias is not None:
                m.bias.data.zero_()
    return


def model_parameter_init_xavier_uniform(model_):
    for m in model_.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.05)
            if m.bias is not None:
                m.bias.data.zero_()
    return


def model_parameter_init_kaiming(model_):
    for m in model_.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)


class BaseModel:
    def __init__(self, my_model, name=None, checkpoint_path='', initmode='none', verbose=False):
        if name is not None:
            self.model_name = name
        else:
            self.model_name = str(type(my_model))
        self.path = checkpoint_path
        self.verbose_notify = verbose
        self.records = get_records(self._record_file(), self.verbose_notify)
        self.model = my_model
        self.initmode = initmode
        self.is_training = True
        self.stage = 'init'

    def parameters(self):
        return self.model.parameters()

    def eval(self):
        self.is_training = False
        self.model.eval()
        self.stage = 'eval'

    def train(self):
        self.is_training = True
        self.model.train()
        self.stage = 'train'

    def _get_model_base(self):
        if isinstance(self.model, nn.DataParallel) or isinstance(self.model, DistributedDataParallel):
            m = self.model.module
        else:
            m = self.model
        return m

    def parameters_init(self):
        bm = self._get_model_base()
        if self.initmode == 'none':
            pass
        elif self.initmode == 'kaiming':
            model_parameter_init_kaiming(bm)
        elif self.initmode == 'xavier':
            model_parameter_init_xavier_uniform(bm)
        else:
            model_parameter_init_normal(bm)

    def _record_file(self):
        return os.path.join(self.path, '%s.log' % self.model_name)

    def load(self, count=-1):
        cnt = len(self.records)
        if abs(count) > cnt or count == 0:
            print('Error: invalid checkpoint number %d. Only %d records are saved.' % (count, cnt))
            return
        if count > 0:
            count -= 1
        path = os.path.join(self.path, self.records[count])
        print('Loading checkpoint: %s.' % path)
        load_checkpoint(self._get_model_base(), path)

    def save(self):
        if not os.path.exists(self.path):
            try:
                os.mkdir(self.path)
            except OSError:
                print('Failed to create the target checkpoint path: %s, Aborting.' % self.path)
                return
        name = time.strftime(self.model_name + '_%y-%m-%d_%H:%M:%S.pth')
        print('Saving checkpoint: %s' % name)
        misc.save_to_file(self._record_file(), name)
        save_checkpoint(self._get_model_base(), os.path.join(self.path, name))

    def parameters_count(self):
        m = self._get_model_base()
        return sum(map(lambda x: x.numel(), m.parameters()))

    def report_status(self, notify=False):
        rcd = 'Model: %s, %d parameters, %s stage.' % (self.model_name, self.parameters_count(), self.stage)
        if notify:
            print(rcd)
        return rcd

    def to(self, device):
        self.model.to(device)

    def parallel(self):
        self.model = nn.DataParallel(self.model)

    def single(self):
        self.model = self.model.module

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class LossModel:
    def __init__(self, options, verbose=False):
        self.options = options
        self.verbose_notify = verbose
        self.handle = get_loss(options)
        device, _, _ = probe_gpu(options, verbose)
        self.handle.to(device)
        self.loss = 0
        self.loss_desc = self.handle.desc
        self.loss_desc = ['Whole loss'] + list(self.loss_desc)
        self.collect_cnt = len(self.loss_desc)
        self.loss_collect = [0] * self.collect_cnt
        self.perf = misc.LossesHelper(self.loss_desc)
        self.activated = False
        self._loss = torch.Tensor((0.0, ))

    def __call__(self, component_pairs):
        self.activated = True
        return self.forward(component_pairs)

    @abc.abstractmethod
    def forward(self, component_pairs):
        pass

    def report_status(self, notify=False):
        return self.perf.last_record(notify)

    def flush(self):
        self.perf.flush()

    def report_all_averages(self, notify=False):
        return self.perf.report_all_averages(notify)

    def report_all_std(self, notify=False):
        return self.perf.report_all_std(notify)

    def backward(self):
        if not self.activated:
            raise RuntimeError('Loss calculation is not ready. call %s() before the backward process.'
                               % self.__class__.__name__)
        self.activated = False
        return self._loss.backward()

    def to(self, device):
        self.handle.to(device)


class SingleBaseTester(metaclass=abc.ABCMeta):
    def __init__(self, options, model=None, verbose=True):
        self.verbose_notify = verbose
        self.options = options
        self.device, self.gpu, self.multi_card = probe_gpu(options, verbose)
        if model is None:
            self.model = get_model(options, verbose)
        else:
            self.model = model
        self.model.to(self.device)
        self.model_loc = self.model.path
        if 'model_self_loading' in options.keys() and options['model_self_loading'] is True:
            print('Model is loaded in user-defined style.')
        else:
            if len(self.model.records) == 0:
                raise RuntimeError('Restoring failed. No records found in checkpoint %s.' % self.model_loc)
            else:
                self.model.load()
        self.model.eval()

    def report_status(self, notify=False):
        return self.model.report_status(notify)

    @abc.abstractmethod
    def iter_test(self, train_pairs):
        pass


class SingleBaseTrainer(metaclass=abc.ABCMeta):
    def __init__(self, options, model=None, verbose=True):
        self.last_epoch = 0
        self.last_batch = 0
        self.verbose_notify = verbose
        self.options = options
        if model is not None:
            self.model = model
        else:
            self.model = get_model(options, verbose)
        self.gpu = False
        self.multi_card = False
        self.device, self.gpu, self.multi_card = probe_gpu(options)
        self.model.to(self.device)
        self.adaptive_resume = options['adaptive_resume']
        self.model_loc = self.model.path
        self.model_name = self.model.model_name
        self.records = get_records(self._record_file(), verbose)
        self.model.train()
        self.activated = True
        self.optimizer = make_optimizer(self.options, self.model)
        self.scheduler = make_scheduler(self.options, self.optimizer)
        if self.adaptive_resume:
            if len(self.records) == 0:
                print('Skipped adaptive resuming, no records found in train state %s.' % self._record_file())
                self.adaptive_resume = False
            else:
                self.last_epoch, self.last_batch = self.load_train_state()
        if self.multi_card:
            self.model.parallel()

    def _record_file(self):
        return os.path.join(self.model_loc, '%s_train_states.log' % self.model_name)

    def get_init_lr(self):
        return self.options['learning_rate']

    def get_current_lr(self):
        return self.optimizer.param_groups[-1]['lr']

    def load_train_state(self, count=-1):
        cnt = len(self.records)
        if abs(count) > cnt or count == 0:
            raise RuntimeError('Error: invalid state number %d. Only %d records are saved.' % (count, cnt))
        if count > 0:
            count -= 1
        self.model.load(count)
        path = os.path.join(self.model_loc, self.records[count])
        print('Loading state: %s.' % path)
        state = torch.load(path)
        epoch = state['epoch']
        batch_iter = state['iter']
        opt_state = state['optimizer']
        self.optimizer.load_state_dict(opt_state)
        if self.scheduler is not None:
            sch_state = state['scheduler']
            self.scheduler.load_state_dict(sch_state)
        return epoch, batch_iter

    def save_train_state(self, epoch, batch_iter):
        if not os.path.exists(self.model_loc):
            try:
                os.mkdir(self.model_loc)
            except OSError:
                print('Failed to create the target state path: %s, Aborting.' % self.model_loc)
                return
        if self.scheduler is not None:
            state = {'epoch': epoch,
                     'iter': batch_iter,
                     'scheduler': self.scheduler.state_dict(),
                     'optimizer': self.optimizer.state_dict()}
        else:
            state = {'epoch': epoch,
                     'iter': batch_iter,
                     'optimizer': self.optimizer.state_dict()}
        state_name = self.model_name + '_E%d_B%d' % (epoch, batch_iter)
        name = time.strftime(state_name + '_%y-%m-%d_%H:%M:%S.state')
        print('Saving training state: %s' % name)
        misc.save_to_file(self._record_file(), name)
        save_path = os.path.join(self.model_loc, name)
        torch.save(state, save_path)
        self.model.save()

    def train(self):
        self.model.train()
        self.activated = True

    def eval(self):
        self.model.eval()
        self.activated = False

    def report_status(self, notify=False):
        return self.model.report_status(notify)

    @abc.abstractmethod
    def iter_batch(self, epoch, batch, train_pairs):
        pass

    @abc.abstractmethod
    def batch_update(self, batch, notify=False):
        pass

    @abc.abstractmethod
    def epoch_update(self, epoch, notify=False):
        pass
