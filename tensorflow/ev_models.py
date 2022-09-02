import tensorflow as tf
import netutils as utils
import netblocks as nb
import algorithm as alg
import debug


def eb_net(inp0, name='', is_training=True):
    int_layer = 32
    hist_layers = 64
    act_func = tf.nn.tanh
    sp_act_func = tf.nn.leaky_relu
    final_act = tf.nn.tanh

    lbl = tf.reduce_max(inp0, axis=-1, keepdims=True)
    tvm = alg.get_saturation(inp0)
    inp = tf.concat([lbl, tvm], axis=-1)
    with tf.variable_scope(name):
        with tf.variable_scope('histogram_priors'):
            hist, _, _ = nb.hist_layer(inp, hist_layers, training=False, hist_type='global',
                                           name='hist%d' % hist_layers)
            lat_ch = utils.get_channels(hist)
            hist = tf.layers.dense(hist, lat_ch, activation=act_func, name='fc1')
        with tf.variable_scope('spatial_priors'):
            spat1 = tf.layers.conv2d(inp0, int_layer, 3, strides=(2, 2), padding='VALID',
                                     activation=sp_act_func, use_bias=False)
            spat2 = tf.layers.conv2d(inp0, int_layer, 5, strides=(2, 2), padding='VALID',
                                     activation=sp_act_func, use_bias=False)
            spat2 = tf.pad(spat2, [[0, 0], [0, 1], [0, 1], [0, 0]], mode='REFLECT')
            spat = tf.concat([spat1, spat2], axis=-1)
            spat = tf.layers.conv2d(spat, int_layer * 2, 3, strides=(2, 2), padding='VALID',
                                     activation=sp_act_func, use_bias=False)
            spat = tf.layers.conv2d(spat, int_layer * 4, 3, strides=(2, 2), padding='VALID',
                                    activation=sp_act_func, use_bias=False)
            spat = tf.layers.conv2d(spat, int_layer * 8, 3, strides=(2, 2), padding='VALID',
                                    activation=sp_act_func, use_bias=False)
            spat = utils.batch_norm(spat, is_training, name='spat_norm')
            spat = tf.reduce_mean(spat, axis=[1, 2], keepdims=False)
        out = tf.concat([spat, hist], axis=-1)
        debug.global_debug_cache['EB_latents'] = out
        out = tf.layers.dense(out, 120, activation=act_func, name='out_fc1')
        out = tf.layers.dense(out, 20, activation=act_func, name='out_fc2')
        out = tf.layers.dense(out, 1, activation=final_act, name='final_fc')
        return out

