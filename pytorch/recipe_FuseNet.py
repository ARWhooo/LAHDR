from default_config import DEFAULT_TEST_ARGUMENTATIONS, DEFAULT_TRAIN_ARGUMENTATIONS
from model.EnhanceNet import EnhanceNet
from model.EBNet import GlobalLightRectification, EBNet
from model.FuseNet import get_multi_images
from network import load_checkpoint
import numpy as np
import torch
from utils.misc import save_to_file, torch2cv
import utils.image_io as iio
import os


ENHANCENET_CKPT_DIR = './trained_parameters/EnhanceNet.pth'
EBNET_CKPT_DIR = './trained_parameters/EBNet.pth'
TEST_TYPE = 'LDR'   # 'LDR_UE', 'LDR_OE'
configs = {
    'model_name': 'FuseNet_Heads3_G1.5_B16D3_Shared+Concat_NoBias+Prelu_lr1e-4_L1+10Color+10TVL-L1_Train780x4',
    'model_parameters': {'heads': 3, 'in_ch': 3, 'base_ch': 16, 'depth': 3, 'seperate': False,
                         'mode': 'concat', 'bias': False, 'act': 'prelu'},
    'model': 'FuseNet',
    'batch_size': 16,
    'epochs': 200,
    'decay_type': 'step',
    'sch_decay_step': 5,
    'sch_decay_gamma': 0.9,
    'model_initmode': 'xavier',
    'loss': 'IntegratedLoss',
    'loss_parameters': {'loss_types': ['L1', 'Color', 'TVL_L1'], 'loss_weights': [1.0, 10.0, 10.0]},
    'learning_rate': 1e-4,
    'gpu_ids': '0',
    'procedure_handle': 'recipe_FuseNet',
    'save_handle': 'recipe_FuseNet',
    'dataset': {
            'train': {
                'dataset_type': 'H5KVDataset',
                'data_source': '/data/DML_LVZ_HDR_Train_780.h5',
                'fetch_keys': ['HDR', 'LDR'],
                'exclude_keys': [],
                'argumentations': DEFAULT_TRAIN_ARGUMENTATIONS
            },
            'test': {
                'dataset_type': 'H5FetchDataset',
                'data_source': '/data/LDR-HDR-Pairs_Test.h5',
                'fetch_keys': ['HDR', TEST_TYPE],
                'argumentations': DEFAULT_TEST_ARGUMENTATIONS
            }
    },
    'test_interval': 50,
    'save_interval': 50,
    'metrics': ['PQ-PSNR', 'VDP2'],
    # 'num_workers': 0
}


GAMMA_1 = 1.5
GAMMA_2 = 1.5
enhance_net = EnhanceNet(3, gf_kernel=5, int_layers=32, layers=4, ksize=3)
eb_net = EBNet(int_layers=32, hist_layers=64)
offseter = GlobalLightRectification(mu=200, factor=3.0, norm_enable=False)
load_checkpoint(enhance_net, ENHANCENET_CKPT_DIR)
load_checkpoint(eb_net, EBNET_CKPT_DIR)


def generate_train_samples(x):
    with torch.no_grad():
        enh = torch.clamp(enhance_net(x), 0, 1)
        eb = eb_net(x)
        oft, eb = offseter(enh, eb)
        return get_multi_images(oft, GAMMA_1, GAMMA_2), eb


def train_process(train_pairs, model, loss_handle, device, need_grad=True):
    hdr = train_pairs[0]
    ldr = train_pairs[1]
    hdr = torch.clamp(hdr, 0, 1)
    ldr = torch.clamp(ldr, 0, 1)
    ldr = ldr.to(device)
    hdr = hdr.to(device)
    eb_net.to(device)
    enhance_net.to(device)
    with torch.no_grad():
        inp, eb = generate_train_samples(ldr)
    if need_grad:
        pred = model(inp)
        loss = loss_handle((pred, hdr))
    else:
        with torch.no_grad():
            pred = model(inp)
            loss = loss_handle((pred, hdr))
    return (pred, eb), loss


def eval_process(epoch, tester, predictor, evaluater, losshelper, result, save_loc, log_file):
    for batch, (hdr, ldr) in enumerate(tester):
        rcd = 'Test batch %d: ' % (batch + 1)
        pred, _ = predictor((hdr, ldr))
        pred, eb = pred
        eb = eb.squeeze().item()
        hdr = torch.clamp(hdr, 0, 1)
        ldr = torch.clamp(ldr, 0, 1)
        pred = torch2cv(pred.squeeze())
        hdr = torch2cv(hdr.squeeze())
        ldr = torch2cv(ldr.squeeze())
        rcd += evaluater.process(None, pred, hdr)
        if result is not None:
            result(save_loc, epoch, batch + 1, eb, pred, ldr, hdr)
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


def result(dir, epoch, batch, eb, pred, ldr, hdr):
    if not os.path.exists(dir):
        try:
            os.mkdir(dir)
        except OSError:
            print('Failed to create the save path: %s, Aborting.' % dir)
            return
    if not os.path.exists(os.path.join(dir, 'E%d' % epoch)):
        os.mkdir(os.path.join(dir, 'E%d' % epoch))
    name = 'B%03d' % batch
    iio.save_HDR(os.path.join(dir, 'E%d' % epoch, 'predict_' + name + '.exr'), pred)
    iio.save_HDR(os.path.join(dir, 'E%d' % epoch, 'origin_' + name + '.exr'), hdr)
    iio.save_LDR(os.path.join(dir, 'E%d' % epoch, 'input_' + name + '.jpg'), np.clip(ldr, 0, 1) * 255.0)
    save_to_file(os.path.join(dir, 'E%d' % epoch, 'Exposure_Biases.txt'), 'Bias for batch %03d: %.4f.' % (batch, eb))

