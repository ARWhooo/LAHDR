import os
import tensorflow as tf
import numpy as np
import loss as ls
import utils as utl
import image_io as iio
import dataloader as dl
import algorithm as alg
from network import Model, fusenet
import merge_models as mmdl


model_name = 'FuseNet'
checkpoint_dir = './checkpoints/%s' % model_name
test_save_dir = './results/%s' % model_name
train_dir = '/data/DML_LVZ_HDR_Train_780.h5'
test_dir = '/data/LDR-HDR-Pairs_Test.h5'
sample_log_train = './checkpoints/%s/train_samples.log' % model_name
sample_log_test = './checkpoints/%s/train_test.log' % model_name
exclude_datasrc = []
base_name = 'Merge_HDR'
loss_type = ['L1', 'Color', 'TVL_L1']
loss_weights = [1.0, 10.0, 10.0]  # VGG weight, TV weight
base_mdl = fusenet(mmdl.fuse_net, checkpoint_dir, base_name)
base_mdl.fuse_gamma = 1.5
patchsize = 256
epochs = 200
batchsize = 16
learning_rate = 1e-4
print_train_result = False
adaptive_resume = True
test_interval = 20
test_save_epochs = [20, 120, 200]
checkpoint_interval = 20
train_decay = True
decay_epochs = 20
all_fetches_train = ['HDR', 'RAW']
all_fetches_test = ['HDR', 'RAW']
tf.set_random_seed(200)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
DEFAULT_GAMMA = 2.2
transfer_type = 'none'
transfer_mu_value = 5000.0
enable_vdp2 = True

if transfer_type == 'mu':
    loss_transfer = lambda x: alg.mu_law_inverse(x, mu=transfer_mu_value)
    out_transfer = lambda x: alg.mu_law_forward(x, mu=transfer_mu_value)
    print('All loss terms have been transferred in mu-%.1f space.' % transfer_mu_value)
elif transfer_type == 'pq':
    loss_transfer = alg.pq_inverse
    out_transfer = alg.pq_forward
    print('All loss terms have been transferred in PQ space.')
else:
    loss_transfer = lambda x: x
    out_transfer = lambda x: x
    print('No transfers have been applied.')


def loss_function(output_, output_pic):
    output_pic = loss_transfer(output_pic)
    loss, loss_collect, loss_desc = ls.get_loss_function(output_, output_pic, loss_type, loss_weights)
    return loss, loss_collect, loss_desc


def log10_hist(inp, dir):
    inp[inp < 1e-6] = 1e-6
    pred_H = np.log10(inp)
    iio.draw_hist(pred_H, -6, 1, 1024, dir, name='Log10 Histogram')


def save_results(contents, test_save_dir, epoch, bt):
    pred = np.squeeze(contents[0])
    gt = np.squeeze(contents[1])
    inp_pic = np.squeeze(contents[2])
    if epoch in test_save_epochs:
        iio.save_HDR(os.path.join(test_save_dir, 'E%d_B%d_pred.exr' % (epoch, bt)), np.squeeze(pred))
        iio.save_HDR(os.path.join(test_save_dir, 'E%d_B%d_orig.exr' % (epoch, bt)), np.squeeze(gt))
        iio.save_HDR(os.path.join(test_save_dir, 'E%d_B%d_inpt.tiff' % (epoch, bt)), inp_pic * 65535.0)
        log10_hist(pred, os.path.join(test_save_dir, 'E%d_B%d_predHist.jpg' % (epoch, bt)))
        log10_hist(gt, os.path.join(test_save_dir, 'E%d_B%d_gtHist.jpg' % (epoch, bt)))
    if enable_vdp2:
        vdp_logger.log(pred, gt, epoch, bt)


def map_func_train(contents):
    sz = len(contents[all_fetches_train[0]])
    sample_cache = np.zeros((sz, patchsize, patchsize, 3), dtype=np.float32)
    label_cache = np.zeros((sz, patchsize, patchsize, 3), dtype=np.float32)
    for i in range(sz):
        ldr = contents[all_fetches_train[1]][i].astype(np.float32)
        hdr = contents[all_fetches_train[0]][i].astype(np.float32)
        label_cache[i] = hdr
        sample_cache[i] = ldr
    return sample_cache, label_cache


def map_func_test(contents):
    return np.expand_dims(contents[all_fetches_test[1]][0], axis=0).astype(np.float32), \
           np.expand_dims(contents[all_fetches_test[0]][0], axis=0).astype(np.float32)


def get_datasets():
    d = dl.SelectedKVLoader(train_dir, batchsize, epochs, all_fetches_train, map_func_train, exclude_datasrc,
                            sample_log_train, shuffle=True)
    t = dl.SelectedKVLoader(test_dir, 1, 1, all_fetches_test, map_func_test, exclude_datasrc,
                            sample_log_test, shuffle=False)
    return d, t


def feeder(testset, epoch):
    ldr, gt = testset.iter_batch()
    feeds = {output_pic: gt, input_pic: ldr}
    return feeds


def metrics(inp, lbl, maxval=1.0):
    inp = alg.pq_inverse(inp)
    lbl = alg.pq_inverse(lbl)
    return tf.reduce_mean(tf.image.psnr(inp, lbl, maxval))


if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
    adaptive_resume = False
if not os.path.exists(test_save_dir):
    os.mkdir(test_save_dir)
procname = os.path.basename(checkpoint_dir)
savefile_name = os.path.join(checkpoint_dir, procname + '.txt')
dataset, testset = get_datasets()

if enable_vdp2:
    import vdp2
    vdp_logger = vdp2.VDPLogger(checkpoint_dir, testset.get_totalsize())

dsize = None
input_pic = tf.placeholder(dtype=tf.float32, shape=[None, dsize, dsize, 3], name='LDR_inp')
output_pic = tf.placeholder(dtype=tf.float32, shape=[None, dsize, dsize, 3], name='merged_out')
evaluations = save_results

# Module Process
output_ = base_mdl(input_pic, is_training=True)
perfor = metrics(out_transfer(output_), output_pic)
loss, loss_collect, loss_desc = loss_function(output_, output_pic)
loss_collect.append(perfor)
loss_desc.append('PQ-PSNR')
contents = [out_transfer(output_), output_pic, input_pic]

# Training Parameters
global_step = tf.Variable(0, trainable=False, name='global_step')
if train_decay:
    totals = dataset.total_size
    decay_step = totals // batchsize * decay_epochs
    print('Adaptive decay step configured as %d (%d epochs).' % (decay_step, decay_epochs))
else:
    decay_step = 0
base_mdl.optim(loss, learning_rate, train_decay, decay_step, contents, loss_collect, loss_desc)

# GPU configuration for training
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # Do not canibalize the entire GPU

# Training
with tf.Session(config=config) as sess:
    flags = utl.single_net_train_flags(savefile_name, test_interval, adaptive_resume, print_train_result,
                                       test_save_dir, checkpoint_interval)
    utl.single_net_train(sess, base_mdl, None, feeder, evaluations, dataset, testset, flags)
