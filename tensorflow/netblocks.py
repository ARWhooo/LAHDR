import tensorflow as tf
import netutils as utils
import numpy as np
from collections import OrderedDict


def init_histlayer_params(inp, bins_per_channel, minval=0.0, maxval=1.0, training=True):
    inp_ch = inp.get_shape().as_list()[-1]
    bins_count = inp_ch * bins_per_channel
    # Default bin centers and widths for initializing histogram
    init_bin_width = (maxval - minval) / bins_per_channel
    bin_init_center = [(minval + k * init_bin_width + init_bin_width / 2) for k in range(bins_per_channel)]
    if training:
        cent_conv_bias = tf.get_variable('hist_centers',
                                         initializer=np.array(bin_init_center * inp_ch).astype(np.float32),
                                         dtype=tf.float32)
        width_conv_weight = tf.get_variable('hist_widths', [1, 1, bins_count, 1],
                                            initializer=tf.constant_initializer(init_bin_width),
                                            dtype=tf.float32)
    else:
        cent_conv_bias = tf.constant(np.array(bin_init_center * inp_ch).astype(np.float32),
                                     dtype=tf.float32, name='hist_centers')
        width_conv_weight = tf.constant(init_bin_width, shape=[1, 1, bins_count, 1], dtype=tf.float32,
                                        name='hist_widths')
    return cent_conv_bias, width_conv_weight


def layer_hist(inp, bins_per_channel, minval=0.0, maxval=1.0, normalize=True, training=True, params=None):
    inp_ch = inp.get_shape().as_list()[-1]
    if params is None:
        cent_conv_bias, width_conv_weight = init_histlayer_params(inp, bins_per_channel, minval, maxval, training)
    else:
        cent_conv_bias = params[0]
        width_conv_weight = params[1]
        o_ch = utils.get_shape(cent_conv_bias)[0]
        bins_per_channel = o_ch // inp_ch
    temp = 0
    for i in range(inp_ch):
        for j in range(bins_per_channel):
            if i == 0 and j == 0:
                temp = tf.expand_dims(inp[:, :, :, i], axis=-1)
            else:
                temp = tf.concat([temp, tf.expand_dims(inp[:, :, :, i], axis=-1)], axis=-1)
    it = temp
    # added biases representing centers for each of the inp_ch*bins_per_channel bins
    it = tf.nn.bias_add(it, (-1.0) * cent_conv_bias)
    # convolution weights representing widths for each of the inp_ch*bins_per_channel bins
    it = tf.nn.depthwise_conv2d(it, 1.0 / width_conv_weight, strides=[1, 1, 1, 1], padding='SAME')
    # Modulate with radical basis function for differentiable binning
    it = tf.exp(-tf.square(it)) + tf.constant(1e-6)
    # Normalize the bins
    if normalize:
        norm = 0
        for i in range(inp_ch):
            it_scale = tf.reduce_sum(it[:, :, :, (i * bins_per_channel):((i + 1) * bins_per_channel)], axis=3, keepdims=True)
            if i == 0:
                norm = it[:, :, :, (i * bins_per_channel):((i + 1) * bins_per_channel)] / it_scale
            else:
                tmp = it[:, :, :, (i * bins_per_channel):((i + 1) * bins_per_channel)] / it_scale
                norm = tf.concat([norm, tmp], axis=-1)
        #it = it / scale
        it = norm
    return it, cent_conv_bias, width_conv_weight


def hist_layer(inp, bins_per_channel, minval=0.0, maxval=1.0, training=True, name='hist_layer', hist_type='global',
               local_patchsize=64, modulate=True, params=None):
    inp_ch = utils.get_channels(inp)

    with tf.variable_scope(name):
        if hist_type == 'global':
            it, cent_conv_bias, width_conv_weight = layer_hist(inp, bins_per_channel, minval, maxval, True, training,
                                                               params)
            hist = tf.reduce_mean(it, axis=[1, 2], keepdims=False)
        else:
            it, cent_conv_bias, width_conv_weight = layer_hist(inp, bins_per_channel, minval, maxval, True, training,
                                                               params)
            if modulate:
                for i in range(inp_ch):
                    if i == 0:
                        norm = tf.expand_dims(inp[:, :, :, i], axis=-1) * it[:, :, :, (i*bins_per_channel):((i+1)*bins_per_channel)]
                    else:
                        tmp = tf.expand_dims(inp[:, :, :, i], axis=-1) * it[:, :, :, (i*bins_per_channel):((i+1)*bins_per_channel)]
                        norm = tf.concat([norm, tmp], axis=-1)
                it = norm
            if local_patchsize == 1:
                hist = it
            else:
                if hist_type == 'local_dense':
                    hist = tf.nn.avg_pool(it, [1, local_patchsize, local_patchsize, 1], [1, 1, 1, 1], padding='VALID')
                else:
                    hist = tf.nn.avg_pool(it, [1, local_patchsize, local_patchsize, 1],
                                          [1, local_patchsize, local_patchsize, 1],
                                          padding='VALID')
    return hist, cent_conv_bias, width_conv_weight


def depthwise_conv(inp, ksize, out_scale, use_bias=False, name='depthwise_conv', activation=None, mode='REFLECT',
                   summary=False):
    inp_ch = utils.get_channels(inp)
    with tf.variable_scope(name):
        w = utils.weight_variable([ksize, ksize, inp_ch, out_scale], name='dc_weight')
        if ksize > 1:
            inp = tf.pad(inp, [[0, 0], [ksize // 2, ksize // 2], [ksize // 2, ksize // 2], [0, 0]], mode=mode)
        inp = tf.nn.depthwise_conv2d(inp, w, [1, 1, 1, 1], padding='VALID', name='dc_conv')
        if summary:
            tf.summary.histogram('weight', tf.reshape(w, [-1]))
        if use_bias:
            b = utils.bias_variable([inp_ch * out_scale], name='dc_bias')
            inp = tf.nn.bias_add(inp, b, name='dc_bias_add')
            if summary:
                tf.summary.histogram('bias', tf.reshape(b, [-1]))
        if activation is not None:
            inp = activation(inp, name='dc_act')
        return inp


# Dense block architecture
# input bottleneck function type should be: func(inp, kernel_size, out_layernum, name, summary)
def dense_block(inp, bottleneck, nb_layers, ksizes, layer_kernum, name, summary=False):
    with tf.variable_scope(name):
        if len(ksizes) != nb_layers or len(layer_kernum) != nb_layers:
            print('Error: Dense block requires kernel size and layer number params for each of the layers.')
            exit(-1)
        layers = [inp]

        inp0 = bottleneck(inp, ksizes[0], layer_kernum[0], 'layer0', summary=summary)
        layers.append(inp0)

        for i in range(nb_layers-1):
            inp1 = tf.concat(layers, axis=3)
            inp1 = bottleneck(inp1, ksizes[i+1], layer_kernum[i+1], 'layer%d' % (i+1), summary=summary)
            layers.append(inp1)

        return tf.concat(layers, axis=3)


def hist_transfer_block(inp, bins_per_channel, minval=0.0, maxval=1.0, training=True, name='hist_block', params=None,
                        modulate=True, activation=tf.nn.leaky_relu, trans_kernels=None):
    if trans_kernels is None:
        trans_kernels = [1, 3, 1]

    with tf.variable_scope(name):
        inp_ch = utils.get_channels(inp)
        inp, center, width = hist_layer(inp, bins_per_channel, minval, maxval, training, name='hist_layer',
                                        hist_type='local', local_patchsize=1, modulate=modulate, params=params)
        org_inp = inp
        for i in range(len(trans_kernels)):
            ksize = trans_kernels[i]
            inp = depthwise_conv(inp, ksize, 1, use_bias=True, name='layer%d' % (i+1), activation=activation)
        out = inp + org_inp

        final = 0
        for j in range(inp_ch):
            slice = out[:, :, :, j*bins_per_channel:(j+1)*bins_per_channel]
            slice = tf.reduce_sum(slice, axis=-1, keepdims=True)
            if j == 0:
                final = slice
            else:
                final = tf.concat([final, slice], axis=-1)
        return final, center, width

