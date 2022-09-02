import os
import image_io as iio
import dataloader as dl
import utils as utl
import numpy as np
import tensorflow as tf
import enhance_models as hmdl
import ev_models as emdl
from network import Model, ebnet
import loss as ls
import time


dsize = None
eb_name = 'EB_Predict'
model_name = 'EBNet'
checkpoint_dir = './checkpoints/%s' % model_name
test_save_dir = './results/%s' % model_name
train_dir = '/data/EV_Adobe_Train_+RAW.h5'
test_dir = '/data/EV_Pairs_ldr.h5'
input_type = 'hdr'
output_type = 'hdr'
epochs = 200
batchsize = 16
learning_rate = 1e-3
print_train_result = False
adaptive_resume = True
test_interval = 50
checkpoint_interval = 50
train_decay = True
decay_epochs = 20
all_fetches_train = ['-2', '-1', '0', '1', '2']
all_fetches_test = ['-1', '0', '1']
all_pairs_test = [['1', '0'], ['-1', '0']]
all_pairs_train = [['-2', '1'], ['-1', '1'], ['0', '1'], ['2', '1']]
transfer_mu_value = 200.0
transfer_norm_enable = False
eb_multiply_basis = 3.0
eb_file = os.path.join(checkpoint_dir, 'EB_MU%.1f_%s_Basis%.1f.txt' %
                       (transfer_mu_value, 'Normed' if transfer_norm_enable else 'Clipped', eb_multiply_basis))
tf.set_random_seed(200)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
DEFAULT_GAMMA = 2.2


def bright_level_loss(inp, lbl, type='l1'):
    eps = 1e-6
    inp_hm = tf.reduce_mean(tf.log(inp + eps), axis=[1, 2, 3])
    lbl_hm = tf.reduce_mean(tf.log(lbl + eps), axis=[1, 2, 3])
    if type == 'l1':
        delta = tf.abs(tf.exp(inp_hm) - tf.exp(lbl_hm))
    else:
        delta = tf.square(tf.exp(inp_hm) - tf.exp(lbl_hm))
    return tf.reduce_mean(delta)


def mean_level_loss(inp, lbl, type='l1'):
    eps = 1e-6
    inp_hm = tf.reduce_mean(inp, axis=[1, 2, 3])
    lbl_hm = tf.reduce_mean(lbl, axis=[1, 2, 3])
    if type == 'l1':
        delta = tf.abs(inp_hm - lbl_hm)
    else:
        delta = tf.square(inp_hm - lbl_hm)
    return tf.reduce_mean(delta)


def train_loader(epoch):
    return loss_desc, loss_collect, train_op, loss, contents


def feeder(testset, epoch):
    ldr, gt = testset.iter_batch()
    feeds = {input_pic: ldr, output_pic: gt}
    return feeds


def map_func_test(contents):
    pairs = all_pairs_test[np.random.randint(len(all_pairs_test))]
    sample_cache = np.stack(contents[pairs[0]], axis=0) / 255.0
    label_cache = np.stack(contents[pairs[1]], axis=0) / 255.0
    return sample_cache, label_cache


def map_func_train(contents):
    pairs = all_pairs_train[np.random.randint(len(all_pairs_train))]
    sample_cache = np.stack(contents[pairs[0]], axis=0)
    label_cache = np.stack(contents[pairs[1]], axis=0)
    return sample_cache, label_cache


def save_results(contents, test_save_dir, epoch, bt):
    pred = contents[0]
    gt = contents[1]
    inp_pic = contents[2]
    eb = contents[3]
    save_img(os.path.join(test_save_dir, 'E%d_B%d_pred.tiff' % (epoch, bt)), output_type, pred)
    save_img(os.path.join(test_save_dir, 'E%d_B%d_orig.tiff' % (epoch, bt)), output_type, gt)
    save_img(os.path.join(test_save_dir, 'E%d_B%d_inpt.tiff' % (epoch, bt)), input_type, inp_pic)
    rcd_str = 'Epoch %d, Batch %d: Predicted EV: %.3f.' % (epoch, bt, np.squeeze(eb))
    utl.save_to_file(eb_file, rcd_str)


def metrics(inp, lbl, maxval=65535.0):
    inp = inp * 65535.0
    lbl = lbl * 65535.0
    return tf.reduce_mean(tf.image.psnr(inp, lbl, maxval))


def save_img(loc, type_, img):
    if type_ == 'ldr':
        save_normalized_ldr(loc, img)
    else:
        save_normalized_hdr(loc, img)


def save_normalized_ldr(loc, img):
    inpt_ldr = np.squeeze(np.clip(np.power(img, 1 / DEFAULT_GAMMA), 0, 1)) * 255
    iio.save_LDR(loc, inpt_ldr)


def save_normalized_hdr(loc, img):
    inpt_ldr = np.squeeze(np.clip(img, 0, 1)) * 65535
    iio.save_HDR(loc, inpt_ldr)


def get_datasets():
    d = dl.SingleDataFromH5(train_dir, batchsize, epochs, all_fetches_train, map_func_train, shuffle=True)
    t = dl.SingleDataFromH5(test_dir, 1, 1, all_fetches_test, map_func_test, shuffle=False)
    return d, t


if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
    adaptive_resume = False
if not os.path.exists(test_save_dir):
    os.mkdir(test_save_dir)
procname = os.path.basename(checkpoint_dir)
savefile_name = os.path.join(checkpoint_dir, procname + '.txt')
dataset, testset = get_datasets()
evaluations = save_results

input_pic = tf.placeholder(dtype=tf.float32, shape=[None, dsize, dsize, 3], name='input')
output_pic = tf.placeholder(dtype=tf.float32, shape=[None, dsize, dsize, 3], name='output')
enhnet = Model(hmdl.enhance_net, './checkpoints/EnhanceNet',
               'LDR_Enhance')
ebpnet = ebnet(emdl.eb_net, checkpoint_dir, eb_name)
ebpnet.multi_factor = eb_multiply_basis
ebpnet.mu_value = transfer_mu_value
ebpnet.norm_enable = transfer_norm_enable
enh_inp = enhnet(input_pic, False)
enh_ref = enhnet(output_pic, False)
oft, eb = ebpnet(input_pic, enh_inp, True)

b_loss = bright_level_loss(oft, enh_ref, 'l1')
loss = b_loss
loss_collect = [b_loss]
loss_desc = ['Bright level loss']
contents = [oft, enh_ref, enh_inp, eb]
perfor = metrics(oft, enh_ref)
loss_collect.append(perfor)
loss_desc.append('PSNR')

# Training Parameters
global_step = tf.Variable(0, trainable=False, name='global_step')
if train_decay:
    totals = dataset.total_size
    decay_step = totals // batchsize * decay_epochs
    print('Adaptive decay step configured as %d (%d epochs).' % (decay_step, decay_epochs))
    start_lr = tf.train.exponential_decay(learning_rate, global_step, decay_step, 0.9, staircase=True)
else:
    start_lr = tf.convert_to_tensor(learning_rate)
g_vars, var_all = utl.get_train_save_parameters(eb_name)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = tf.train.AdamOptimizer(start_lr).minimize(loss, global_step=global_step, var_list=g_vars)
saver = tf.train.Saver(var_all)

# GPU configuration for training
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # Do not canibalize the entire GPU

# Training
with tf.Session(config=config) as sess:
    _ = sess.run(tf.global_variables_initializer())
    enhnet.restore(sess)
    ebpnet.status()
    step = 0
    if adaptive_resume:
        epoch, batch, global_step = utl.load_latest_state(saver, sess, checkpoint_dir,
                                                          dataset.get_total_batchnum(), global_step)
    else:
        epoch = 1
        batch = 1
    sess.graph.finalize()
    try:
        while epoch <= dataset.get_total_epochs():
            loss_dec, loss_collect, train_op, loss, train_contents = train_loader(epoch)
            lossHelp = utl.LossesHelper(loss_dec)
            mean_loss = []
            mean_perf = []
            train_time = []
            while batch <= dataset.get_total_batchnum():
                st_time = time.time()
                feeds = feeder(dataset, epoch)
                whole_loss, _, lossvals = sess.run([loss, train_op, loss_collect], feed_dict=feeds)
                if print_train_result:
                    rcd_str = 'Training: epoch %d, batch %d: whole loss: %f.' % \
                              (epoch, batch, whole_loss)
                    print(rcd_str)
                    lossHelp.iter_record(lossvals, True)
                else:
                    lossHelp.iter_record(lossvals, False)
                mean_loss.append(whole_loss)
                batch += 1
                train_time.append(time.time() - st_time)
            m_loss = np.mean(np.array(mean_loss))
            m_time = np.mean(np.array(train_time))
            rcd_str = 'Mean loss for epoch %d: %f, average time: %f, ' % (epoch, m_loss, m_time)
            print(rcd_str)
            utl.save_to_file(savefile_name, rcd_str + lossHelp.report_all_averages())
            mean_loss.clear()
            lossHelp.flush()
            if epoch % test_interval == 0:
                bt = 1
                while bt <= testset.get_total_batchnum():
                    feeds = feeder(testset, epoch)
                    output, stage_losses, stage_collects = sess.run([train_contents, loss, loss_collect],
                                                                            feed_dict=feeds)
                    rcd_str = 'Testing epoch %d, batch %d: whole loss: %f. ' % (epoch, bt, stage_losses)
                    print(rcd_str)
                    rcd_str += lossHelp.iter_record(stage_collects, True)
                    utl.save_to_file(savefile_name, rcd_str)
                    evaluations(output, test_save_dir, epoch, bt)
                    mean_loss.append(stage_losses)
                    bt += 1
                m_loss = np.mean(np.array(mean_loss))
                rcd_str = 'Mean loss for whole epoch: %f. ' % (m_loss)
                print(rcd_str)
                utl.save_to_file(savefile_name, rcd_str + lossHelp.report_all_averages())
                mean_loss.clear()
                lossHelp.flush()
            if epoch % checkpoint_interval == 0:
                saver.save(sess, os.path.join(checkpoint_dir, 'on_test.ckpt'), global_step)
            epoch += 1
            batch = 1
    except KeyboardInterrupt:
        print("Whole training complete. %d epochs trained." % (epoch - 1))

    chkpt_path = os.path.join(checkpoint_dir, 'on_stop.ckpt')
    saver.save(sess, save_path=chkpt_path)
