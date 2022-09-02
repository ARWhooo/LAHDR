import os, time
import importlib
import argparse
import torch
import numpy as np
from dataset import get_dataset
from utils import get_procedure_handle, get_save_handle, get_logfile, MetricEvaluation, get_logger
import default_config
from network import ImagePairTrainer, ImagePairTester
from network import flop_status


PRINT_FLOPS = False
PRINT_PARAMETERS_COUNT = True


def checkpoint_reset(opt):
    dir_ = opt['checkpoint_dir']
    name = opt['model_name']
    save = opt['save_location']
    if not os.path.exists(dir_):
        os.mkdir(dir_)
    if not os.path.exists(os.path.join(dir_, name)):
        os.mkdir(os.path.join(dir_, name))
    if not os.path.exists(save):
        os.mkdir(save)
    if not os.path.exists(os.path.join(save, name)):
        os.mkdir(os.path.join(save, name))
    opt['checkpoint_dir'] = os.path.join(dir_, name)
    opt['save_location'] = os.path.join(save, name)
    opt['model_name'] = opt['model']
    return opt


def need_test(epoch, opt):
    if 'test_epochs' in opt.keys():
        if epoch in opt['test_epochs']:
            return True
        else:
            return False
    else:
        return epoch % opt['test_interval'] == 0


def need_save(epoch, opt):
    if 'save_epochs' in opt.keys():
        if epoch in opt['save_epochs']:
            return True
        else:
            return False
    else:
        return epoch % opt['save_interval'] == 0


def train(opt):
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    seed = 200
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    opt = checkpoint_reset(opt)
    train_process, eval_process = get_procedure_handle(opt)
    trainer = ImagePairTrainer(opt, train_process, None, False)
    log_file = get_logfile(opt, 'train')
    logger = get_logger('default').set_loc(log_file)
    if PRINT_FLOPS:
        flop_status(trainer.model.model)
    else:
        if PRINT_PARAMETERS_COUNT:
            logger('', trainer.report_status(False), display=True)
    start_epoch = trainer.last_epoch + 1
    loader = get_dataset(opt, 'train')
    tester = get_dataset(opt, 'test')
    epochs = opt['epochs']
    losshelper = trainer.report_handle()

    evaluater = MetricEvaluation(opt)
    result = get_save_handle(opt)
    save_loc = opt['save_location']
    cur_epoch = 1

    try:
        for epoch in range(start_epoch, epochs + 1):
            st_time = time.time()
            for batch, source in enumerate(loader):
                trainer.iter_batch(epoch, batch, source)
                rcd = 'Epoch %d, batch %d: ' % (epoch, batch + 1)
                rcd += trainer.batch_update(batch + 1, False)
                if opt['print_iter_result']:
                    logger('', rcd, display=True)
            ed_time = time.time() - st_time
            logger('Average for epoch %d' % epoch, losshelper.report_all_averages(False), display=True)
            print('Duration: %.4f.' % ed_time)
            losshelper.flush()
            trainer.epoch_update(epoch, False)
            cur_epoch = epoch
            if need_save(epoch, opt):
                trainer.save_train_state(epoch, 1)
            if need_test(epoch, opt):
                logger('', 'Current lr: %.6f.' % trainer.get_current_lr(), display=True)
                trainer.eval()
                eval_process(epoch, tester, trainer.prediction, evaluater, losshelper, result, save_loc, log_file)
                trainer.train()
    except KeyboardInterrupt:
        print('Training aborted by user.')
    finally:
        print('Training completed. %d epoch(s) trained.' % cur_epoch)


def evaluate(opt):
    opt = checkpoint_reset(opt)
    train_process, eval_process = get_procedure_handle(opt)
    tester = ImagePairTester(opt, train_process, None, True)
    loader = get_dataset(opt, 'test')
    evaluater = MetricEvaluation(opt)
    losshelper = tester.report_handle()
    log_file = get_logfile(opt, 'evaluate')
    save_loc = opt['save_location']
    logger = get_logger('default').set_loc(log_file)
    result = get_save_handle(opt)
    if PRINT_FLOPS:
        flop_status(tester.model.model)
    else:
        if PRINT_PARAMETERS_COUNT:
            logger('', tester.report_status(False), display=True)
    eval_process(1, loader, tester.iter_test, evaluater, losshelper, result, save_loc, log_file)


def get_args():
    recipe = 'recipe_LAHDR'
    stage = 'test'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=recipe, help='The specific .py config file in the '
                                                                   'current folder.')
    parser.add_argument('--stage', type=str, default=stage, help='State among "train" and "evaluate".')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    args = get_args()
    opt = default_config.DefaultConfig()
    params = importlib.import_module(args.config)
    if hasattr(params, 'configs'):
        extra_opt = params.configs
    else:
        extra_opt = {}
        print('No "configs" dict existed in file: %s' % args.config)
        exit(-1)
    options = opt.parse(extra_opt, True)
    options['stage'] = args.stage
    print('Selected stage: %s' % args.stage)
    if args.stage == 'train':
        train(options)
    else:
        evaluate(options)
