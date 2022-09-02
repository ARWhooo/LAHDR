import tensorflow as tf
import guided_filter as gf
import netutils as utils
import netblocks as nb
import debug
from collections import OrderedDict
import algorithm as alg


def conv_concat(inp, ks, chs, name, bias, act, norm=None):
    with tf.variable_scope(name):
        if act is not None:
            inp = act(inp, name='activation')
        for i in range(len(ks) - 1):
            inp = utils.pad_conv(inp, ks[i], chs[i], 'conv%d' % (i + 1), None, bias)
        inp = utils.pad_conv(inp, ks[-1], chs[-1], 'conv%d' % len(ks), None, False)
        if norm is not None:
            inp = norm(inp, name='out_norm')
    return inp


def feature_masks(inputs, mask_trainable):
    with tf.variable_scope('mask_generation'):
        mask_main = fusion_mask(inputs[0], 'mask_main', mask_trainable)
        mask_pic2 = fusion_mask(inputs[1], 'mask_pic2', mask_trainable)
        mask_pic3 = fusion_mask(inputs[2], 'mask_pic3', mask_trainable)
        masks = nb.depthwise_conv(tf.concat([mask_main, mask_pic2, mask_pic3], axis=-1), 3, 1, use_bias=True,
                                  name='mask_transfer', activation=None)
        masks = tf.nn.softmax(masks, axis=-1)
    return masks


def get_evLevel(rgbimg, mu=0.5, sigma=0.2):
    dem = 2 * sigma * sigma
    img = rgbimg
    out = -1.0 * tf.square(img - mu)
    return tf.exp(out / dem)


def fusion_mask(inp, name, trainable=False):
    with tf.variable_scope(name):
        lum = alg.get_luminance(inp)
        t = alg.get_gradient(lum)
        s = alg.get_saturation(inp)
        if trainable:
            mu = tf.get_variable('exposure_mean', shape=(1), dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.5))
        else:
            mu = 0.5
        e = get_evLevel(lum, mu, 0.2)
        if trainable:
            p = tf.concat([t, s, e], axis=-1)
            m = utils.pad_conv(p, 1, 3, 'att_conv1', activation=None)
            m = utils.pad_conv(m, 1, 1, 'att_conv2', activation=tf.nn.sigmoid)
        else:
            m = t * s * e
        return m


def mask_global_all(inputs, masks, itl, depth, out_channel, out_shape, seperate_enc, base_transfer_enc,
                    dw_op, use_bias, lat_norm=True, is_training=True):
    with tf.variable_scope('global_branch', reuse=tf.AUTO_REUSE):
        glb1 = tf.image.resize_bilinear(inputs[0], (128, 128), name='resize_main')
        glb2 = tf.image.resize_bilinear(inputs[1], (128, 128), name='resize_pic2')
        glb3 = tf.image.resize_bilinear(inputs[2], (128, 128), name='resize_pic3')
        masks = tf.image.resize_bilinear(masks, (128, 128), name='resize_mask')
        msks = [masks[:, :, :, :1], masks[:, :, :, 1:2], masks[:, :, :, 2:3]]
        z = itl
        for d in range(depth):
            if seperate_enc:
                glb1 = base_transfer_enc(glb1, [3], [z], 'layer_dw%d_main' % (d + 1), use_bias)
                glb2 = base_transfer_enc(glb2, [3], [z], 'layer_dw%d_pic2' % (d + 1), use_bias)
                glb3 = base_transfer_enc(glb3, [3], [z], 'layer_dw%d_pic3' % (d + 1), use_bias)
            else:
                glb1 = base_transfer_enc(glb1, [3], [z], 'layer_dw%d_shared' % (d + 1), use_bias)
                glb2 = base_transfer_enc(glb2, [3], [z], 'layer_dw%d_shared' % (d + 1), use_bias)
                glb3 = base_transfer_enc(glb3, [3], [z], 'layer_dw%d_shared' % (d + 1), use_bias)
            glb1 = dw_op(glb1, 2, d + 1)
            glb2 = dw_op(glb2, 2, d + 1)
            glb3 = dw_op(glb3, 2, d + 1)
            masks = utils.max_pool(masks, 2)
            z = z * 2
        sz = out_shape
        one = tf.convert_to_tensor((1,), dtype=sz.dtype)
        shp = tf.concat([one, sz, one], axis=0)
        glb_ch = out_channel
        if lat_norm:
            glb1 = utils.batch_norm(glb1, is_training, name='norm_main')
            glb2 = utils.batch_norm(glb2, is_training, name='norm_pic2')
            glb3 = utils.batch_norm(glb3, is_training, name='norm_pic3')
        glb1 = tf.reduce_mean(glb1 * masks[:, :, :, :1], axis=[1, 2], keepdims=False)
        glb2 = tf.reduce_mean(glb2 * masks[:, :, :, 1:2], axis=[1, 2], keepdims=False)
        glb3 = tf.reduce_mean(glb3 * masks[:, :, :, 2:], axis=[1, 2], keepdims=False)
        glb = tf.concat([glb1, glb2, glb3], axis=-1)
        glb = tf.layers.dense(glb, glb_ch, tf.nn.sigmoid, use_bias=True, name='fc1')
        glb = tf.tile(tf.expand_dims(tf.expand_dims(glb, axis=1), axis=1), shp)
    return glb


def fuse_net(inputs, name='', is_training=True):
    itl = 16
    depth = 3
    out_ch = utils.get_channels(inputs[0])
    dec_merge_type = 'concat'
    base_act = tf.nn.leaky_relu
    use_bias = False
    seperate_enc = False
    mask_trainable = True
    base_conv_2 = conv_concat
    norm = None  # lambda x, name: utils.instance_norm(x, is_training, name=name)
    base_transfer_enc = lambda inp, ks, chs, name, bias: base_conv_2(inp, ks, chs, name, bias, base_act, norm=norm)
    base_transfer_dec = lambda inp, ks, chs, name, bias: base_conv_2(inp, ks, chs, name, bias, base_act, norm=norm)
    latent_transfer = lambda inp, ks, chs, name, bias: base_conv_2(inp, ks, chs, name, bias, base_act, norm=norm)
    up_op = lambda inp, size, i: tf.image.resize_bilinear(inp, size, name='bilinear%d' % i)
    dw_op = lambda inp, dw_size, i: utils.conv_2d(inp, 3, utils.get_channels(inp), 2, 'down_conv%d' % i)

    with tf.variable_scope(name):
        layers = OrderedDict({0: OrderedDict(), 1: OrderedDict(), 2: OrderedDict()})
        masks = feature_masks(inputs, mask_trainable)
        sizes = OrderedDict()
        main = inputs[0]
        pic2 = inputs[1]
        pic3 = inputs[2]
        debug.global_debug_cache['histout'] = main
        k = itl
        glb_dw_op = lambda x, d, s: utils.max_pool(x, d, name='maxpool%d' % s)
        with tf.variable_scope('encoders', reuse=tf.AUTO_REUSE):
            for d in range(depth):
                if seperate_enc:
                    main = base_transfer_enc(main, [3], [k], 'layer_dw%d_main' % (d + 1), use_bias)
                    pic2 = base_transfer_enc(pic2, [3], [k], 'layer_dw%d_pic2' % (d + 1), use_bias)
                    pic3 = base_transfer_enc(pic3, [3], [k], 'layer_dw%d_pic3' % (d + 1), use_bias)
                else:
                    main = base_transfer_enc(main, [3], [k], 'layer_dw%d_shared' % (d + 1), use_bias)
                    pic2 = base_transfer_enc(pic2, [3], [k], 'layer_dw%d_shared' % (d + 1), use_bias)
                    pic3 = base_transfer_enc(pic3, [3], [k], 'layer_dw%d_shared' % (d + 1), use_bias)
                sizes[d + 1] = utils.get_size(main)
                layers[0][d + 1] = main * masks[:, :, :, :1]
                layers[1][d + 1] = pic2 * masks[:, :, :, 1:2]
                layers[2][d + 1] = pic3 * masks[:, :, :, 2:]
                main = dw_op(main, 2, d + 1)
                pic2 = dw_op(pic2, 2, d + 1)
                pic3 = dw_op(pic3, 2, d + 1)
                masks = tf.nn.max_pool(masks, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='mask_dw%d' % (d + 1))
                k = k * 2
        lat_ch = utils.get_channels(main)
        lat_shp = utils.get_size(main)
        glb = mask_global_all(inputs, masks, 16, 3, lat_ch, lat_shp, seperate_enc, base_transfer_enc,
                              glb_dw_op, use_bias, lat_norm=True, is_training=is_training)
        latent = main * masks[:, :, :, :1] + pic2 * masks[:, :, :, 1:2] + pic3 * masks[:, :, :, 2:]
        latent = latent_transfer(latent, [3, 3], [lat_ch, lat_ch], 'latent_block1', use_bias)
        latent = tf.concat([latent, glb], axis=-1)
        latent = latent_transfer(latent, [3, 3], [lat_ch, lat_ch], 'latent_block2', use_bias)
        merge_skips = OrderedDict()
        dw_layers = OrderedDict()
        for j in range(1, depth + 1):
            merge_skips[j] = layers[0][j] + layers[1][j] + layers[2][j]
        with tf.variable_scope('decoder'):
            for d in range(depth, 0, -1):
                latent = up_op(latent, sizes[d], d)
                l = utils.get_channels(merge_skips[d])
                if dec_merge_type == 'mid_add':
                    latent = base_transfer_dec(latent, [3], [l], 'layer_up%d_1' % d, use_bias)
                    latent += merge_skips[d]
                    latent = base_transfer_dec(latent, [3], [l], 'layer_up%d_2' % d, use_bias)
                elif dec_merge_type == 'add':
                    latent = base_transfer_dec(latent, [3, 3], [l, l], 'layer_up%d' % d, use_bias)
                    latent += merge_skips[d]
                else:
                    latent = base_transfer_dec(latent, [3], [l], 'layer_up%d_1' % d, use_bias)
                    latent = tf.concat([latent, merge_skips[d]], axis=-1)
                    latent = base_transfer_dec(latent, [3], [l], 'layer_up%d_2' % d, use_bias)
                dw_layers[d] = latent
        temp = []
        for t in range(1, depth + 1):
            it = dw_layers[t]
            dw_layers[t] = utils.pad_conv(it, 3, out_ch, name='output%d' % t, activation=tf.nn.relu, use_bias=False)
            temp.append(dw_layers[t])
        return dw_layers, temp

