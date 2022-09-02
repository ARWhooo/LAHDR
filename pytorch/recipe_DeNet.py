import numpy as np
import torch, os
from default_config import DEFAULT_TEST_ARGUMENTATIONS
from utils.misc import save_to_file, torch2cv
import utils.image_io as iio
from utils.raw_noise import raw_random_noise


NOISE_SCHEME = 'SIDD'
TEST_ON_REAL_DATASET = True
if TEST_ON_REAL_DATASET:
    test_data_loc = '/data/SIDD_RAW.h5'
    fetch_keys = ['GT', 'Noise']
else:
    test_data_loc = '/data/EV_Adobe_Test_+RAW.h5'
    fetch_keys = ['RAW']


configs = {
    'model_name': 'DeNet_I32_L5_lr1e-4_L1_Adobe5K-S1',
    'model_parameters': {'in_ch': 3, 'out_ch': 3, 'int_layers': 32, 'layers': 5},
    'model': 'DeNet',
    'batch_size': 32,
    'epochs': 120,
    'decay_type': 'none',
    'lr_decay': 0.95,
    'weight_decay': 1e-4,
    'num_workers': 20,
    'gpu_ids': '0',
    'save_interval': 50,
    'test_interval': 50,
    # 'model_initmode': 'normal',
    'loss': 'IntegratedLoss',
    'loss_parameters': {'loss_types': ['l1'], 'loss_weights': [1.0]},
    'learning_rate': 1e-4,
    'metrics': ['PSNR'],
    'procedure_handle': 'recipe_DeNet',
    'save_handle': 'recipe_DeNet',
    'dataset': {
            'train': {
                'dataset_type': 'H5FetchDataset',
                'data_source': '/data/EV_Adobe_Train_+RAW.h5',
                'fetch_keys': ['RAW'],
                'argumentations': DEFAULT_TEST_ARGUMENTATIONS
            },
            'test': {
                'dataset_type': 'H5FetchDataset',
                'data_source': test_data_loc,
                'fetch_keys': fetch_keys,
                'argumentations': DEFAULT_TEST_ARGUMENTATIONS
            }
    },
}


def train_process(train_pairs, model, loss_handle, device, need_grad=True):
    if need_grad:
        hdr = train_pairs
        hdr = torch.clamp(hdr / 65535.0, 0, 1)
        ldr = torch.clamp(raw_random_noise(hdr, NOISE_SCHEME), 0, 1)
        ldr = ldr.to(device)
        hdr = hdr.to(device)
        pred = model(ldr)
        loss = loss_handle((pred, hdr))
    else:
        if TEST_ON_REAL_DATASET:
            hdr, ldr = train_pairs
            hdr = torch.clamp(hdr / 65535.0, 0, 1)
            ldr = torch.clamp(ldr / 65535.0, 0, 1)
        else:
            hdr = train_pairs
            hdr = torch.clamp(hdr / 65535.0, 0, 1)
            ldr = torch.clamp(raw_random_noise(hdr, NOISE_SCHEME), 0, 1)
        ldr = ldr.to(device)
        hdr = hdr.to(device)
        with torch.no_grad():
            pred = model(ldr)
            loss = loss_handle((pred, hdr))
    return (pred, hdr), loss


def eval_process(epoch, tester, predictor, evaluater, losshelper, result, save_loc, log_file):
    for batch, content in enumerate(tester):
        rcd = 'Test batch %d: ' % (batch + 1)
        pred, _ = predictor((content, ))
        pred, hdr = pred
        pred = torch2cv(pred.squeeze())
        hdr = torch2cv(hdr.squeeze())
        pred = np.clip(pred, 0, 1)
        rcd += evaluater.process(None, hdr, pred)
        if result is not None:
            result(save_loc, epoch, batch + 1, pred, hdr)
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


def result(dir, epoch, batch, pred, hdr):
    if not os.path.exists(dir):
        try:
            os.mkdir(dir)
        except OSError:
            print('Failed to create the save path: %s, Aborting.' % dir)
            return
    if not os.path.exists(os.path.join(dir, 'E%d' % epoch)):
        os.mkdir(os.path.join(dir, 'E%d' % epoch))
    name = 'B%03d' % batch
    hdr = hdr * 65535
    pred = pred * 65535
    iio.save_HDR(os.path.join(dir, 'E%d' % epoch, 'PRED_' + name + '.tiff'), pred)
    iio.save_HDR(os.path.join(dir, 'E%d' % epoch, 'GT_' + name + '.tiff'), hdr)
