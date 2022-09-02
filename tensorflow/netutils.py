import tensorflow as tf


def prelu(inp, name='prelu'):
    alpha = tf.Variable(initial_value=0.2, dtype=tf.float32, name=name + '_params')
    return tf.nn.leaky_relu(inp, alpha=alpha, name=name)


def layer_norm(inp, is_training=False, name='layer_norm'):
    epsilon = 1e-5
    with tf.variable_scope(name):
        mean, var = tf.nn.moments(inp, [0], keep_dims=True)
        scale = tf.get_variable(
            'scale', [get_channels(inp)],
            initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02), dtype=inp.dtype
        )
        offset = tf.get_variable(
            'offset', [get_channels(inp)], initializer=tf.constant_initializer(0.0),
            dtype=inp.dtype
        )
        if is_training:
            return scale * tf.div(inp - mean, tf.sqrt(var + epsilon)) + offset
        else:
            return tf.div(inp - mean, tf.sqrt(var + epsilon))


def instance_norm(inp, is_training=True, name='instance_norm'):
    epsilon = 1e-5
    with tf.variable_scope(name):
        mean, var = tf.nn.moments(inp, [1, 2], keep_dims=True)
        if is_training:
            scale = tf.get_variable(
                'scale', [get_channels(inp)],
                initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02), dtype=inp.dtype
            )
            offset = tf.get_variable(
                'offset', [get_channels(inp)], initializer=tf.constant_initializer(0.0),
                dtype=inp.dtype
            )
            return scale * tf.div(inp - mean, tf.sqrt(var + epsilon)) + offset
        else:
            return tf.div(inp - mean, tf.sqrt(var + epsilon))


#From: https://github.com/taki0112/GCNet-Tensorflow/blob/master/GCNet.py
def spectral_norm(w, iteration=1, name='spectral_norm'):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    with tf.variable_scope(name):
        u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

        u_hat = u
        v_hat = None
        for i in range(iteration):
            """
            power iteration
            Usually iteration = 1 will be enough
            """
            v_ = tf.matmul(u_hat, tf.transpose(w))
            v_hat = tf.nn.l2_normalize(v_)

            u_ = tf.matmul(v_hat, w)
            u_hat = tf.nn.l2_normalize(u_)

        u_hat = tf.stop_gradient(u_hat)
        v_hat = tf.stop_gradient(v_hat)

        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = w / sigma
            w_norm = tf.reshape(w_norm, w_shape)

        return w_norm


def weight_variable(shape, stddev=0.1, type='xavier', name="weight"):
    if type == 'xavier':
        initial = tf.contrib.layers.xavier_initializer(uniform=True)
    else:
        initial = tf.truncated_normal_initializer(stddev=stddev)
    weight_regularizer = tf.contrib.layers.l2_regularizer(0.00001)
    return tf.get_variable(name, shape=shape, initializer=initial, regularizer=weight_regularizer)


def bias_variable(shape, name="bias"):
    weight_regularizer = tf.contrib.layers.l2_regularizer(0.00001)
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(0), regularizer=weight_regularizer)


def pad_conv(inp, ksize, channels, name='pad_conv', activation=None, use_bias=False, type='REFLECT', summary=False):
    with tf.variable_scope(name):
        pad_size = ksize // 2
        pad_inp = tf.pad(inp, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], mode=type)
        out = conv_2d(pad_inp, ksize, channels, 1, name=name, padding='VALID', use_bias=use_bias,
                      activation=activation, summary=summary)
        return out


def dilate_conv(inp, ksize, dilation, channels, name='dilate_conv', activation=None, use_bias=False,
                type='REFLECT', summary=False):
    with tf.variable_scope(name):
        pad_size = (ksize + 2 * (dilation - 1)) // 2
        pad_inp = tf.pad(inp, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], mode=type)
        inp_channel = get_shape(inp)[-1]
        W = weight_variable([ksize, ksize, inp_channel, channels], name='weight')
        if summary:
            tf.summary.histogram('weight', tf.reshape(W, [-1]))
        conv = tf.nn.atrous_conv2d(pad_inp, W, dilation, padding='VALID')
        if use_bias:
            b = bias_variable([channels], name='bias')
            if summary:
                tf.summary.histogram('bias', tf.reshape(b, [-1]))
            conv = tf.nn.bias_add(conv, b)
        if activation is not None:
            return activation(conv, name='activation')
        else:
            return conv


def conv_2d(x, ksize, out_channel, stride, name, use_bias=False,
            padding='SAME', init_std=0.1, activation=None, summary=False):
    with tf.variable_scope(name):
        inp_channel = get_shape(x)[-1]
        inp_shape = [ksize, ksize, inp_channel, out_channel]
        strides = [1, stride, stride, 1]
        W = weight_variable(inp_shape, stddev=init_std, name='weight')
        if summary:
            tf.summary.histogram('weight', tf.reshape(W, [-1]))
        conv = tf.nn.conv2d(x, W, strides=strides, padding=padding)
        if use_bias:
            b = bias_variable([out_channel], name='bias')
            if summary:
                tf.summary.histogram('bias', tf.reshape(b, [-1]))
            conv = tf.nn.bias_add(conv, b)
        if activation is not None:
            return activation(conv, name='activation')
        else:
            return conv


def hw_flatten(x):
    batchnum = tf.shape(x)[0:1]
    ch = tf.convert_to_tensor(get_channels(x))
    ch = tf.reshape(ch, (1, ))
    tmp = tf.convert_to_tensor(-1, dtype=ch.dtype)
    tmp = tf.reshape(tmp, (1, ))
    shape_t = tf.concat([batchnum, tmp, ch], axis=-1)
    return tf.reshape(x, shape=shape_t)


def deconv_2d(x, ksize, out_channel, stride, out_shape, name, use_bias=False,
            padding='SAME', init_std=0.1, activation=None, summary=False):
    with tf.variable_scope(name):
        if out_shape.get_shape().as_list()[0] == 2:
            out_shape = tf.concat([tf.shape(x)[0:1], out_shape, tf.convert_to_tensor([out_channel])], axis=0)
        inp_channel = get_shape(x)[-1]
        inp_shape = [ksize, ksize, out_channel, inp_channel]
        strides = [1, stride, stride, 1]
        W = weight_variable(inp_shape, stddev=init_std, name='weight')
        if summary:
            tf.summary.histogram('weight', tf.reshape(W, [-1]))
        conv = tf.nn.conv2d_transpose(x, W, out_shape, strides=strides, padding=padding)
        if use_bias:
            b = bias_variable([out_channel], name='bias')
            if summary:
                tf.summary.histogram('bias', tf.reshape(b, [-1]))
            conv = tf.nn.bias_add(conv, b)
        if activation is not None:
            return activation(conv, name='activation')
        else:
            return conv


def max_pool(x, n, padding='SAME', name='max_pool'):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding=padding, name=name)


def avg_pool(x, n, padding='SAME', name='avg_pool'):
    return tf.nn.avg_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding=padding, name=name)


def get_image_summary(img, ch_idx=0, batch_idx=0):
    """
    Make an image summary for 4d tensor image with index idx
    """
    V = tf.slice(img, (batch_idx, 0, 0, ch_idx), (1, -1, -1, 1))
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255
    return V


def get_color_image_summary(img, st_ch_idx=0, batch_idx=0):
    """
    Make an image summary for 4d tensor image with index idx
    """

    V = tf.slice(img, (batch_idx, 0, 0, st_ch_idx), (1, -1, -1, 3))
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255
    return V


def image_channel_summary(img, name_prefix, st_ch_idx=0, batch_idx=0):
    inp_ch = get_channels(img)
    for i in range(st_ch_idx, inp_ch):
        tf.summary.image(name_prefix + '_%d' % i, get_image_summary(img, ch_idx=i, batch_idx=batch_idx))


def scalar_summary(vector, name_prefix, st_idx=0, batch_idx=0):
    inp_ch = get_channels(vector)
    vec = tf.squeeze(vector)
    for i in range(st_idx, inp_ch):
        tf.summary.scalar(name_prefix + '_%d' % i, vec[batch_idx, i:(i + 1)])


def get_shape(x):
    if 'tensorflow' in str(type(x)):
        return x.get_shape().as_list()
    else:
        return list(x.shape)


def get_channels(x):
    if 'tensorflow' in str(type(x)):
        return x.get_shape().as_list()[-1]
    else:
        return list(x.shape)[-1]


def get_size(x):
    if 'tensorflow' in str(type(x)):
        return tf.shape(x)[1:3]
    else:
        return list(x.shape)[1:3]


def activation(inp, name, activate_func=tf.nn.relu):
    return activate_func(inp, name=name)


def batch_norm(inp, is_training, name, reuse=None):
    return tf.layers.batch_normalization(
                    inp,
                    momentum=0.95,
                    epsilon=1e-5,
                    training=is_training,
                    reuse=reuse,
                    name=name)


def gradient(input_tensor, direction, norm='raw'):
    smooth_kernel_x = tf.reshape(tf.constant([[0, 0], [-1, 1]], tf.float32), [2, 2, 1, 1])
    smooth_kernel_y = tf.transpose(smooth_kernel_x, [1, 0, 2, 3])

    if direction == "x":
        kernel = smooth_kernel_x
    else:
        kernel = smooth_kernel_y
    kernel = tf.concat([kernel] * input_tensor.get_shape()[3], axis=2)
    if norm == 'l1':
        return tf.abs(tf.nn.depthwise_conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))
    elif norm == 'l2':
        return tf.square(tf.nn.depthwise_conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))
    else:
        return tf.nn.depthwise_conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME')

