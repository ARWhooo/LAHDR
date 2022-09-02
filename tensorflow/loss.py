import tensorflow as tf
import netutils as utils
import netblocks as nb


def get_loss_function(inp, lbl, loss_types, loss_weights):
    loss = 0
    loss_collect = []
    loss_desc = []
    if isinstance(loss_types, str):
        loss_types = [loss_types]
        loss_weights = [loss_weights]
    for w in range(len(loss_types)):
        t = loss_types[w]
        weight = loss_weights[w]
        if t == 'l1' or t == 'L1':
            ls = content_loss(inp, lbl, type='l1')
            desc = 'L1 loss'
        elif t == 'l2' or t == 'L2':
            ls = content_loss(inp, lbl, type='l2')
            desc = 'L2 loss'
        elif t == 'Color' or t == 'color':
            ls = color_loss(inp, lbl)
            desc = 'Color loss'
        elif t == 'TVL_L1' or t == 'tvl_l1':
            ls = tv_loss(inp, lbl, norm='l1')
            desc = 'L1 TV loss'
        elif t == 'TVL_L2' or t == 'tvl_l2':
            ls = tv_loss(inp, lbl, norm='l2')
            desc = 'L2 TV loss'
        elif t == 'TVR_L1' or t == 'tvr_l1':
            ls = tv_regulation(inp, norm='l1')
            desc = 'L1 TV regulation'
        elif t == 'TVR_L2' or t == 'tvr_l2':
            ls = tv_regulation(inp, norm='l2')
            desc = 'L2 TV regulation'
        elif t == 'ssim' or t == 'SSIM':
            ls = 1.0 - tf.reduce_mean(tf.image.ssim(inp, lbl, max_val=1.0))
            desc = 'SSIM loss'
        elif t == 'mssim' or t == 'MSSIM':
            ls = 1.0 - tf.reduce_mean(tf.image.ssim_multiscale(inp, lbl, max_val=1.0))
            desc = 'MSSIM loss'
        else:
            raise ValueError('Unknown loss type: %s.' % t)
        loss += weight * ls
        loss_collect.append(ls)
        loss_desc.append(desc)
        print('%s applied, weight %.3f.' % (desc, weight))
    return loss, loss_collect, loss_desc


def mssim_loss(inp, ref, maxval=1.0):
    return 1.0 - tf.image.ssim_multiscale(inp, ref, maxval)


def tv_loss(ldr, hdr, norm='l1'):
    loss_h = utils.gradient(ldr, 'x') - utils.gradient(hdr, 'x')
    loss_w = utils.gradient(ldr, 'y') - utils.gradient(hdr, 'y')
    if norm == 'l1':
        result = tf.reduce_mean(0.5 * (tf.abs(loss_h) + tf.abs(loss_w)))
    else:
        result = tf.reduce_mean(0.5 * (tf.square(loss_h) + tf.square(loss_w)))
    return result


def tv_regulation(inp, norm='l2'):
    if norm == 'l2':
        result = tf.reduce_mean(utils.gradient(inp, 'x', norm='l2') + utils.gradient(inp, 'y', norm='l2'))
    else:
        result = tf.reduce_mean(utils.gradient(inp, 'x', norm='l1') + utils.gradient(inp, 'y', norm='l1'))
    return 0.5 * result


def content_loss(hdr, hdr_, type='l1'):
    if type == 'l1':
        return tf.reduce_mean(tf.abs(hdr - hdr_))
    else:
        return tf.reduce_mean(tf.square(hdr - hdr_))


# Copied from DeepUPE
def color_loss(image, label):
    vec1 = tf.reshape(image, [-1, 3])
    vec2 = tf.reshape(label, [-1, 3])
    norm_vec1 = tf.nn.l2_normalize(vec1, 1)
    norm_vec2 = tf.nn.l2_normalize(vec2, 1)
    return tf.losses.cosine_distance(norm_vec1, norm_vec2, axis=1)


def color_constancy_loss(out):
    mean_out = tf.reduce_mean(out, axis=[1, 2])
    loss_const = tf.reduce_mean(((mean_out[:, 0] - mean_out[:, 1]) ** 2 + (mean_out[:, 1] - mean_out[:, 2]) ** 2 +
                                 (mean_out[:, 0] - mean_out[:, 2]) ** 2) / 3.0)
    return loss_const

