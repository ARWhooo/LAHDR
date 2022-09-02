import os
import image_io as iio
import utils as utl
import numpy as np
import tensorflow as tf
import enhance_models as hmdl
import ev_models as emdl
import merge_models as mmdl
from network import Model, ebnet, fusenet


gamma = 1.5
eb_shift = True
denoise = False
inp_dir = '../Samples/GT_LDR_HDRCNN'
outlog = '../LA-HDR_EB_DML.log'
out_dir = '../Ours'

dsize = None
input_pic = tf.placeholder(dtype=tf.float32, shape=[1, dsize, dsize, 3], name='input')
enhnet = Model(hmdl.enhance_net,
               './checkpoints/EnhanceNet',
               'LDR_Enhance')
ebpnet = ebnet(emdl.eb_net,
               './checkpoints/EBNet',
               'EB_Predict')
denet = Model(hmdl.de_net,
              './checkpoints/DeNet',
              'RAW_Denoise')
fusnet = fusenet(mmdl.fuse_net,
                 './checkpoints/FuseNet',
                  'Merge_HDR')
ebpnet.norm_enable = False
fusnet.fuse_gamma = gamma
enh_inp = enhnet(input_pic, False)
if denoise:
    enh_inp = denet(enh_inp, False)
    enh_inp = tf.clip_by_value(enh_inp, 0.0, 1.0)
    print('Candidate low-light denoise model applied.')
if eb_shift:
    oft, ebp = ebpnet(input_pic, enh_inp, False)
    print('Adaptive EB shift applied.')
else:
    print('No EB shift applied.')
    oft = enh_inp / tf.reduce_max(enh_inp)
fus_ = fusnet(oft, False)
out_transfer = lambda x: x#alg.mu_law_inverse(x, 600.0)
fus = out_transfer(fus_)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    enhnet.restore(sess)
    if eb_shift:
        ebpnet.restore(sess)
    if denoise:
        denet.restore(sess)
    fusnet.restore(sess)
    tl = os.listdir(inp_dir)
    tl.sort()
    for j in range(len(tl)):
        a = iio.load_LDR(os.path.join(inp_dir, tl[j])).astype(np.float32) / 255.0
        a = np.expand_dims(a, axis=0)
        if eb_shift:
            ao, eb, Ao = sess.run([oft, ebp, fus], feed_dict={input_pic: a})
            eb = np.squeeze(eb)
            utl.save_to_file(outlog, 'EB for %s: %.4f' % (tl[j], eb))
        else:
            ao, Ao = sess.run([oft, fus], feed_dict={input_pic: a})
        ao = np.squeeze(ao)
        Ao = np.squeeze(Ao)
        iio.save_HDR(os.path.join(out_dir, tl[j].split('.')[0] + '.tiff'), ao * 65535.0)
        iio.save_HDR(os.path.join(out_dir, tl[j].split('.')[0] + '.exr'), Ao)
        print(tl[j])
