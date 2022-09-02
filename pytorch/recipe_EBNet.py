from default_config import DEFAULT_TEST_ARGUMENTATIONS
from model.EnhanceNet import EnhanceNet
from model.EBNet import GlobalLightRectification
from network import load_checkpoint, freeze
import numpy as np
import torch
from utils.misc import save_to_file, torch2cv
import utils.image_io as iio
import os


ENHANCENET_CKPT_DIR = './trained_parameters/EnhanceNet.pth'
configs = {
    'model_name': 'EBNet_Target0_I32+H64_MU200_x3_Clip_lr1e-3_BrightL1Loss_Adobe5k-S1',
    'model_parameters': {'int_layers': 32, 'hist_layers': 64},
    'model': 'EBNet',
    'batch_size': 32,
    'epochs': 300,
    'decay_type': 'step',
    'sch_decay_step': 5,
    'sch_decay_gamma': 0.9,
    # 'lr_decay': 0.95,
    'model_initmode': 'xavier',
    # 'weight_decay': 1e-4,
    'loss': 'EBLoss',
    'loss_parameters': {'loss_type': 'bright', 'norm': 'l1'},
    'learning_rate': 1e-3,
    'gpu_ids': '0',
    'procedure_handle': 'recipe_EBNet',
    'save_handle': 'recipe_EBNet',
    'dataset': {
            'train': {
                'dataset_type': 'ExposureBiasDataset',
                'data_source': '/data/EV_Adobe_Train_+RAW.h5',
                'fetch_keys': ['-2', '-1', '0', '1', '2'],
                'target_idx': 2,
                'argumentations': DEFAULT_TEST_ARGUMENTATIONS
            },
            'test': {
                'dataset_type': 'ExposureBiasDataset',
                'data_source': '/data/EV_Pairs_ldr.h5',
                'target_idx': 1,
                'fetch_keys': ['-1', '0', '1'],
                'argumentations': DEFAULT_TEST_ARGUMENTATIONS
            }
    },
    'test_interval': 50,
    'save_interval': 50,
    'metrics': ['PSNR']
}


enhance_net = EnhanceNet(3, gf_kernel=5, int_layers=32, layers=4, ksize=3)
freeze(enhance_net)
offseter = GlobalLightRectification(mu=200.0, factor=3.0, norm_enable=False)
load_checkpoint(enhance_net, ENHANCENET_CKPT_DIR)


def train_process(train_pairs, model, loss_handle, device, need_grad=True):
    hdr = train_pairs[0]
    ldr = train_pairs[1]
    # hdr = torch.clamp(hdr / 255.0, 0, 1)
    # ldr = torch.clamp(ldr / 255.0, 0, 1)
    enhance_net.to(device)
    ldr = ldr.to(device)
    hdr = hdr.to(device)
    with torch.no_grad():
        ldr_ = torch.clamp(enhance_net(ldr / 255.0), 0, 1)
        hdr_ = torch.clamp(enhance_net(hdr / 255.0), 0, 1)
    if need_grad:
        pred = model(ldr)
        oft, eb = offseter(ldr_, pred)
        loss = loss_handle((oft, hdr_))
    else:
        with torch.no_grad():
            pred = model(ldr)
            oft, eb = offseter(ldr_, pred)
            loss = loss_handle((oft, hdr_))
    return (ldr_, oft, hdr_, eb), loss


def eval_process(epoch, tester, predictor, evaluater, losshelper, result, save_loc, log_file):
    for batch, (hdr, ldr) in enumerate(tester):
        rcd = 'Test batch %d: ' % (batch + 1)
        pred, _ = predictor((hdr, ldr))
        ldr_, oft, hdr_, eb = pred
        ldr_ = torch2cv(ldr_.squeeze())
        oft = torch2cv(oft.squeeze())
        eb = eb.squeeze().item()
        hdr_ = torch2cv(hdr_.squeeze())
        # oft = np.clip(oft, 0, 1)
        # hdr_ = np.clip(hdr_, 0, 1)
        rcd += evaluater.process(None, hdr_, oft)
        if result is not None:
            result(save_loc, epoch, batch + 1, ldr_, eb, oft, hdr_)
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


def result(dir, epoch, batch, ldr, eb, oft, hdr):
    if not os.path.exists(dir):
        try:
            os.mkdir(dir)
        except OSError:
            print('Failed to create the save path: %s, Aborting.' % dir)
            return
    if not os.path.exists(os.path.join(dir, 'E%d' % epoch)):
        os.mkdir(os.path.join(dir, 'E%d' % epoch))
    name = 'B%03d' % batch
    iio.save_HDR(os.path.join(dir, 'E%d' % epoch, 'input_' + name + '.tiff'), ldr * 65535.0)
    iio.save_HDR(os.path.join(dir, 'E%d' % epoch, 'offset_' + name + '.tiff'), oft * 65535.0)
    iio.save_HDR(os.path.join(dir, 'E%d' % epoch, 'origin_' + name + '.tiff'), hdr * 65535.0)
    save_to_file(os.path.join(dir, 'E%d' % epoch, 'Exposure_Biases.txt'), 'Bias for batch %03d: %.4f.' % (batch, eb))
