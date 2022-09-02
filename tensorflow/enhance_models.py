import tensorflow as tf
import netutils as utils
import netblocks as nb
import guided_filter as gf
import debug


def enhance_net(contents, name='', is_training=True):
    filter_kernel_size = 5
    lum = contents
    out = lum - gf.guided_filter(lum, lum, filter_kernel_size)
    int_layer = 32
    layers = 4
    ksize = 3
    summary = False
    out_channels = utils.get_channels(lum) * 2
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        lum1, center1, width1 = nb.hist_transfer_block(lum, int_layer, name='hist_transfer', activation=tf.nn.selu,
                                                       trans_kernels=[1, 3, 3, 1])
        lum1 = tf.nn.relu(lum1)
        debug.global_debug_cache['histout'] = lum1
        inp = tf.concat([lum1, out], axis=-1)
        act_func = tf.nn.leaky_relu
        trans_func = lambda inp, k, ol, name, summary: dense_transfer(inp, k, ol, name,
                                                                      activation=act_func,
                                                                      norm=None,
                                                                      use_bias=False,
                                                                      summary=summary)
        n = layers - 1
        out = nb.dense_block(inp, trans_func, n, [ksize] * n, [int_layer] * n, 'dense_transfer', summary=summary)
        hdrout = dense_transfer(out, ksize, out_channels, 'final_merge', activation=tf.nn.relu,
                                use_bias=False, summary=summary)
        hdr_lum = hdrout[:, :, :, :utils.get_channels(lum)]
        hdr_ref = hdrout[:, :, :, utils.get_channels(lum):]
        hdr_pre = hdr_lum + hdr_ref
        hdr_pre = tf.nn.relu(hdr_pre)
    return hdr_pre


def de_net(contents, name='', is_training=True):
    inp = contents
    int_layer = 32
    layers = 5
    ksize = 3
    summary = False
    out_channels = utils.get_channels(inp)
    with tf.variable_scope(name):
        act_func = tf.nn.leaky_relu
        trans_func = lambda inp, k, ol, name, summary: dilate_multi_conv(inp, k, ol, ol, name, activation=act_func)
        n = layers - 1
        out = nb.dense_block(inp, trans_func, n, [ksize] * n, [int_layer] * n, 'dense_transfer', summary=summary)
        hdrout = dense_transfer(out, ksize, out_channels, 'final_merge', activation=tf.nn.tanh,
                                use_bias=False, summary=summary)
        hdrout = (inp + hdrout) / 2.0
        return hdrout


# norm should be in format: norm(inp, name) if it is not None
def dense_transfer(inp, ksize, ol, name, use_bias=True, activation=tf.nn.relu, norm=None, summary=False):
    if ksize > 1:
        p = ksize // 2
        inp = tf.pad(inp, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
    out = utils.conv_2d(inp, ksize, ol, 1, name, use_bias=use_bias, padding='VALID', activation=activation,
                         summary=summary)
    if norm is not None:
        out = norm(out, name=name+'_norm')
    return out


def dilate_multi_conv(inp, kernel, lat_channels, out_channels, scope, use_bn=None, use_bias=False, activation=None):
    lat = None
    with tf.variable_scope(scope):
        k = kernel
        for d in range(1, kernel + 1):
            it = utils.dilate_conv(inp, k, d, lat_channels, name='dilate_lat_conv%d' % d, activation=activation)
            if lat is None:
                lat = it
            else:
                lat = tf.concat([lat, it], axis=-1)
        if use_bn is not None:
            lat = use_bn(lat, name='lat_norm')
        lat = utils.pad_conv(lat, 1, out_channels, 'merge_conv', use_bias=use_bias, activation=activation)
        skip = utils.conv_2d(inp, 1, out_channels, 1, name='skip_conv')
        return lat + skip

