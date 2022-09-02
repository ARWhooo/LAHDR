import numpy as np
import torch, os
from default_config import DEFAULT_TEST_ARGUMENTATIONS
from utils.misc import save_to_file, torch2cv
import utils.image_io as iio


configs = {
    'model_name': 'EnhanceNet_GF5_I32_L4_lr1e-4_L1_Adobe5K-S1',  # name for this model. Trained parameters are stored in ./checkpoints/[model_name]
    'model_parameters': {'in_ch': 3, 'gf_kernel': 5, 'int_layers': 32, 'layers': 4},  # init parameters to load the exact model (entry 'model' below). Required by nn.Module class [model] in ./model/[model].py 
    'model': 'EnhanceNet',  # model registered in ./model/__init__.py.
    'batch_size': 32,
    'epochs': 300,
    'decay_type': 'none',
    'lr_decay': 0.95,
    'weight_decay': 1e-4,
    'num_workers': 20,
    'gpu_ids': '0',
    'save_interval': 50,
    'test_interval': 50,
    'model_initmode': 'xavier',
    'loss': 'IntegratedLoss',  # loss for training. Loss has to be registered in ./loss/__init__.py 
    'loss_parameters': {'loss_types': ['L1'], 'loss_weights': [1.0]},  # parameters for loading [loss]
    'learning_rate': 1e-4,
    'metrics': ['PSNR'],  # metrics expected to measure in the test / eval stage. See get_metric_handle() function in ./utils/__init__py for details
    'procedure_handle': 'recipe_EnhanceNet',  # python script that stores train_process() and eval_process() functions
    'save_handle': 'recipe_EnhanceNet',  # python script that stores result() function for saving results
    'dataset': {
            'train': {  # required for stage 'train'
                'dataset_type': 'H5FetchDataset',
                'data_source': '/data/EV_Adobe_Train_+RAW.h5',
                'fetch_keys': ['RAW', '0'],
                'argumentations': DEFAULT_TEST_ARGUMENTATIONS
            },
            'test': {  # required for stage 'test'
                'dataset_type': 'H5FetchDataset',
                'data_source': '/data/EV_Adobe_Test_+RAW.h5',
                'fetch_keys': ['RAW', '0'],
                'argumentations': DEFAULT_TEST_ARGUMENTATIONS
            }
    },
}


def train_process(train_pairs, model, loss_handle, device, need_grad=True):
    hdr, ldr = train_pairs
    hdr = torch.clamp(hdr / 65535.0, 0, 1)
    ldr = torch.clamp(ldr / 255.0, 0, 1)
    ldr = ldr.to(device)
    hdr = hdr.to(device)
    if need_grad:
        pred = model(ldr)
        loss = loss_handle((pred, hdr))
    else:
        with torch.no_grad():
            pred = model(ldr)
            loss = loss_handle((pred, hdr))
    return pred, loss


def eval_process(epoch, tester, predictor, evaluater, losshelper, result, save_loc, log_file):
    for batch, (hdr, ldr) in enumerate(tester):
        rcd = 'Test batch %d: ' % (batch + 1)
        pred, _ = predictor((hdr, ldr))
        pred = torch2cv(pred.squeeze())
        ldr = torch2cv(ldr.squeeze())
        hdr = torch2cv(hdr.squeeze())
        ldr = np.clip(ldr / 255.0, 0, 1)
        hdr = np.clip(hdr / 65535.0, 0, 1)
        pred = np.clip(pred, 0, 1)
        rcd += evaluater.process(None, hdr, pred)
        if result is not None:
            result(save_loc, epoch, batch + 1, pred, ldr, hdr)
        print(rcd)
        save_to_file(log_file, rcd)
    rcd_str = 'Test average: '
    print(rcd_str)
    rcd_str += losshelper.report_all_averages(True)
    rcd_str += ' '
    rcd_str += evaluater.report_all_averages(True)
    save_to_file(log_file, rcd_str)
    rcd_str = 'Standard deviation: '
    print(rcd_str)
    rcd_str += losshelper.report_all_std(True)
    rcd_str += ' '
    rcd_str += evaluater.report_all_std(True)
    save_to_file(log_file, rcd_str)
    evaluater.flush()
    losshelper.flush()


def result(dir, epoch, batch, pred, ldr, hdr):
    if not os.path.exists(dir):
        try:
            os.mkdir(dir)
        except OSError:
            print('Failed to create the save path: %s, Aborting.' % dir)
            return
    if not os.path.exists(os.path.join(dir, 'E%d' % epoch)):
        os.mkdir(os.path.join(dir, 'E%d' % epoch))
    name = 'B%03d' % batch
    ldr = ldr * 255
    hdr = hdr * 65535
    pred = pred * 65535
    iio.save_LDR(os.path.join(dir, 'E%d' % epoch, 'INP_' + name + '.jpg'), ldr)
    iio.save_HDR(os.path.join(dir, 'E%d' % epoch, 'PRED_' + name + '.tiff'), pred)
    iio.save_HDR(os.path.join(dir, 'E%d' % epoch, 'GT_' + name + '.tiff'), hdr)
