from default_config import DEFAULT_TEST_ARGUMENTATIONS
import torch
from utils.misc import save_to_file, torch2cv
import utils.image_io as iio
import os


ENHANCENET_CKPT_DIR = './trained_parameters/EnhanceNet.pth'
DENET_CKPT_DIR = './trained_parameters/DeNet.pth'
EBNET_CKPT_DIR = './trained_parameters/EBNet.pth'
FUSENET_CKPT_DIR = './trained_parameters/FuseNet.pth'
NEED_DENOISE = False
LAHDR_parameters = {
    'gamma1': 1.5,
    'gamma2': 1.5,
    'mu': 200,
    'bias_factor': 3.0,
    'bias_norm_enable': False,
    'EnhanceNet': ENHANCENET_CKPT_DIR,
    'EBNet': EBNET_CKPT_DIR,
    'FuseNet': FUSENET_CKPT_DIR,
    'DeNet': DENET_CKPT_DIR
}
TEST_TYPE = 'LDR'   # 'LDR_UE', 'LDR_OE'


# THIS RECIPE IS ONLY PROVIDED FOR THE TESTING SCENARIO.
configs = {
    'model_name': 'LAHDR_Default_GAMMA1-1.5_GAMMA2_1.5',
    'model_parameters': {'need_denoise': NEED_DENOISE, 'trained_configs': LAHDR_parameters},
    'model': 'LAHDR',
    'model_self_loading': True,
    'batch_size': 1,
    'epochs': 1,
    'loss': None,
    'gpu_ids': '0',
    'procedure_handle': 'recipe_LAHDR',
    'save_handle': 'recipe_LAHDR',
    'dataset': {
            'test': {
                'dataset_type': 'H5FetchDataset',
                'data_source': '/data/LDR-HDR-Pairs_Test.h5',
                'fetch_keys': ['HDR', TEST_TYPE],
                'argumentations': DEFAULT_TEST_ARGUMENTATIONS
            }
    },
    'metrics': ['PQ-PSNR', 'VDP2'],   # if you do not want to evaluate VDP2 directly from Python, remove 'VDP2' in this list
}


# Note: the 'need_grad' is 'True' only in the 'train' stage (flag indicated when invoking main.py).
# In the testing scenario (LA-HDR model is only appropriate for the test stage), 'need_grad' is always be 'False'.
def train_process(train_pairs, model, loss_handle, device, need_grad=True):
    ldr = train_pairs
    ldr = torch.clamp(ldr, 0, 1)
    ldr = ldr.to(device)
    if need_grad:
        pred = model(ldr)
        loss = 0
    else:
        with torch.no_grad():
            pred = model(ldr)
            loss = 0
    return pred, loss


def eval_process(epoch, tester, predictor, evaluater, losshelper, result, save_loc, log_file):
    for batch, (hdr, ldr) in enumerate(tester):
        rcd = 'Test batch %d: ' % (batch + 1)
        pred, _ = predictor(ldr)  # execute train_process() with 'need_grad == False'
        pred, eb, mi1, mi2, mi3 = pred
        eb = eb.squeeze().item()
        hdr = torch.clamp(hdr, 0, 1)
        ldr = torch.clamp(ldr, 0, 1)
        mi1 = torch.clamp(mi1, 0, 1)
        mi2 = torch.clamp(mi2, 0, 1)
        mi3 = torch.clamp(mi3, 0, 1)
        pred = torch2cv(pred.squeeze())
        hdr = torch2cv(hdr.squeeze())
        ldr = torch2cv(ldr.squeeze())
        mi1 = torch2cv(mi1.squeeze())
        mi2 = torch2cv(mi2.squeeze())
        mi3 = torch2cv(mi3.squeeze())
        rcd += evaluater.process(None, pred, hdr)  # perform the exact metric evaluations that listed in configs -> metrics
        if result is not None:
            result(save_loc, epoch, batch + 1, eb, mi1, mi2, mi3, pred, ldr, hdr)  # execute result() function that denotes below
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


def result(dir, epoch, batch, eb, mi1, mi2, mi3, pred, ldr, hdr):
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
    iio.save_HDR(os.path.join(dir, 'E%d' % epoch, 'multi-image1_' + name + '.tiff'), mi1 * 65535)
    iio.save_HDR(os.path.join(dir, 'E%d' % epoch, 'multi-image2_' + name + '.tiff'), mi2 * 65535)
    iio.save_HDR(os.path.join(dir, 'E%d' % epoch, 'multi-image3_' + name + '.tiff'), mi3 * 65535)
    iio.save_HDR(os.path.join(dir, 'E%d' % epoch, 'origin_' + name + '.exr'), hdr)
    iio.save_LDR(os.path.join(dir, 'E%d' % epoch, 'input_' + name + '.jpg'), ldr * 255.0)
    save_to_file(os.path.join(dir, 'E%d' % epoch, 'Exposure_Biases.txt'), 'Bias for batch %03d: %.4f.' % (batch, eb))
