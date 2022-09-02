import os
import cv2
import tensorflow as tf
import numpy as np
import enhance_models as hmdl
import loss as ls
import utils as utl
import image_io as iio
import dataloader as dl
import time
import raw_noise as noise


model_name = 'DeNet'
checkpoint_dir = './checkpoints/%s' % model_name
test_save_dir = './results/%s' % model_name
train_dir = '/data/EV_Adobe_Train_+RAW.h5'
test_dir = '/data/EV_Adobe_Test_+RAW.h5'
denoise_name = 'RAW_Denoise'
denoise_mdl = hmdl.de_net
loss_type = ['L1']
loss_weights = [1.0]
patchsize_train = 256
patchsize_test = 256
input_type = 'hdr'
output_type = 'hdr'
noise_type = 'SIDD'
epochs = 200
batchsize = 16
learning_rate = 1e-4
print_train_result = False
adaptive_resume = True
test_interval = 20
checkpoint_interval = 20
train_decay = True
decay_epochs = 20
all_fetches = ['RAW']
tf.set_random_seed(200)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
DEFAULT_GAMMA = 2.2


def train_loader(epoch):
    return loss_desc, loss_collect, train_op, loss, contents


def feeder(testset, epoch, is_training=True):
    if is_training:
        ldr, gt = testset.iter_batch()
        feeds = {input_pic: ldr, output_pic: gt}
    else:
        ldr = testset.iter_batch()
        feeds = {input_pic: ldr}
    return feeds


def map_func_train(contents):
    sz = len(contents[all_fetches[0]])
    sample_cache = np.zeros((sz, patchsize_train, patchsize_train, 3), dtype=np.float32)
    label_cache = np.zeros((sz, patchsize_train, patchsize_train, 3), dtype=np.float32)
    for i in range(sz):
        label = contents[all_fetches[0]][i].astype(np.float32)
        label_cache[i] = label
        sample_cache[i] = np.clip(noise.raw_random_noise(label, noise_type), 0.0, 1.0)
    return sample_cache, label_cache


def map_func_test(contents):
    sz = len(contents[all_fetches[0]])
    sample_cache = np.zeros((sz, patchsize_test, patchsize_test, 3), dtype=np.float32)
    for i in range(sz):
        sample_cache[i] = contents[all_fetches[0]][i][:patchsize_test, :patchsize_test]
    return sample_cache


def save_results(contents, test_save_dir, epoch, bt):
    pred = np.squeeze(contents[0])
    inp_pic = np.squeeze(contents[1])
    save_img(os.path.join(test_save_dir, 'E%d_B%d_pred.tiff' % (epoch, bt)), output_type, pred)
    save_img(os.path.join(test_save_dir, 'E%d_B%d_inpt.tiff' % (epoch, bt)), input_type, inp_pic)


def metrics(inp, lbl, maxval=65535.0):
    inp = inp * maxval
    lbl = lbl * maxval
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
    d = dl.SingleDataFromH5(train_dir, batchsize, epochs, all_fetches, map_func_train, shuffle=True)
    t = dl.SingleDataFromH5(test_dir, 1, 1, all_fetches, map_func_test, shuffle=False)
    return d, t


if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
    adaptive_resume = False
if not os.path.exists(test_save_dir):
    os.mkdir(test_save_dir)
procname = os.path.basename(checkpoint_dir)
savefile_name = os.path.join(checkpoint_dir, procname + '.txt')

dsize = None
input_pic = tf.placeholder(dtype=tf.float32, shape=[None, dsize, dsize, 3], name='input')
output_pic = tf.placeholder(dtype=tf.float32, shape=[None, dsize, dsize, 3], name='enhanced_output')
dataset, testset = get_datasets()
evaluations = save_results

# Module Process
output_ = denoise_mdl(input_pic, denoise_name, is_training=True)
loss, loss_collect, loss_desc = ls.get_loss_function(output_, output_pic, loss_type, loss_weights)
contents = [output_, input_pic]
loss_desc.append('PSNR')
loss_collect.append(metrics(output_, output_pic))

# Training Parameters
global_step = tf.Variable(0, trainable=False, name='global_step')
if train_decay:
    totals = dataset.total_size
    decay_step = totals // batchsize * decay_epochs
    print('Adaptive decay step configured as %d (%d epochs).' % (decay_step, decay_epochs))
    start_lr = tf.train.exponential_decay(learning_rate, global_step, decay_step, 0.9, staircase=True)
else:
    start_lr = tf.convert_to_tensor(learning_rate)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=denoise_name)
with tf.control_dependencies(update_ops):
    train_op = tf.train.AdamOptimizer(start_lr).minimize(loss, global_step=global_step, var_list=g_vars)
list_all = tf.global_variables(scope=denoise_name)
bn_moving_vars = [g for g in list_all if 'moving_mean' in g.name]
bn_moving_vars += [g for g in list_all if 'moving_variance' in g.name]
var_all = g_vars + bn_moving_vars
saver = tf.train.Saver(var_all)

# GPU configuration for training
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # Do not canibalize the entire GPU
#config.gpu_options.per_process_gpu_memory_fraction = 0.2
init_op = tf.global_variables_initializer()

# Training
with tf.Session(config=config) as sess:
    _ = sess.run(init_op)
    utl.print_num_of_total_parameters(denoise_name)
    if adaptive_resume:
        epoch, batch, global_step = utl.load_latest_state(saver, sess, checkpoint_dir,
                                                          dataset.get_total_batchnum(), global_step)
    else:
        epoch = 1
        batch = 1
    sess.graph.finalize()
    loss_dec, loss_collect, train_op, loss, train_contents = train_loader(epoch)
    lossHelp = utl.LossesHelper(loss_dec)
    mean_loss = []
    mean_perf = []
    train_time = []
    try:
        while epoch <= dataset.get_total_epochs():
            while batch <= dataset.get_total_batchnum():
                st_time = time.time()
                feeds = feeder(dataset, epoch, is_training=True)
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
            train_time.clear()
            lossHelp.flush()
            if epoch % checkpoint_interval == 0:
                saver.save(sess, os.path.join(checkpoint_dir, 'on_test.ckpt'), global_step)
            if epoch % test_interval == 0:
                bt = 1
                while bt <= testset.get_total_batchnum():
                    feeds = feeder(testset, epoch, is_training=False)
                    output = sess.run(train_contents, feed_dict=feeds)
                    rcd_str = 'Testing epoch %d, batch %d. ' % (epoch, bt)
                    print(rcd_str)
                    evaluations(output, test_save_dir, epoch, bt)
                    bt += 1
            epoch += 1
            batch = 1
    except KeyboardInterrupt:
        print("Whole training complete. %d epochs trained." % (epoch - 1))

    chkpt_path = os.path.join(checkpoint_dir, 'on_stop.ckpt')
    saver.save(sess, save_path=chkpt_path)
