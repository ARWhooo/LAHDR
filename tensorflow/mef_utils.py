import tensorflow as tf
import numpy as np


def get_enhanced_priors_v1(inp1, gamma=2.0):
    if 'tensorflow' in str(type(inp1)):
        inp2 = 1 - tf.pow(1 - tf.pow(inp1, 1 / (gamma)), (gamma))
        inp3 = tf.pow(1 - tf.pow(1 - inp1, 1 / gamma), gamma)
    else:
        inp2 = 1 - np.power(1 - np.power(inp1, 1 / (gamma)), (gamma))
        inp3 = np.power(1 - np.power(1 - inp1, 1 / gamma), gamma)
    return inp2, inp3


def get_enhanced_priors_v2(inp1, gammas, median_level_2, median_level_3):
    func2 = lambda inp, gamma: (1 - tf.pow(1 - tf.pow(inp, 1 / gamma), gamma))
    func3 = lambda inp, gamma: tf.pow(1 - tf.pow(1 - inp, 1 / (gamma)), gamma)
    mean_func = lambda inp, th: tf.abs(tf.reduce_max(inp, axis=[1, 2, 3], keepdims=True) - th)
    inp2 = adaptive_gamma_selection(inp1, median_level_2, func2, mean_func, gammas)
    inp3 = adaptive_gamma_selection(inp1, median_level_3, func3, mean_func, gammas)
    return inp2, inp3


def adaptive_gamma_selection(inp, mean_th, transfunc, weight_func, gammas):
    #gammas = [1.0, 2.0, 3.0]
    outs = []
    weights = []
    sum = 0
    result = 0
    for gamma in gammas:
        out = transfunc(inp, gamma)
        weight = weight_func(out, mean_th)
        sum += weight
        outs.append(out)
        weights.append(weight)
    count = len(outs)
    for i in range(count):
        result += weights[i] / (sum + 1e-5) * outs[i]
    return result
