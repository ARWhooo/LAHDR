import tensorflow as tf
import utils as utl
import algorithm as alg
import mef_utils as mef
from utils import Model


class ebnet(Model):
    def __init__(self, mdl, mdl_dir, mdl_name):
        super().__init__(mdl, mdl_dir, mdl_name)
        self.multi_factor = 3.0
        self.mu_value = 200.0
        self.norm_enable = True

    def __call__(self, inp, enh, is_training=True):
        self.eb = self.mdl(inp, name=self.name, is_training=is_training)
        self.eb = self.eb * self.multi_factor
        self.training = is_training
        return self._eb_offset(enh, self.eb), self.eb

    def _eb_offset(self, inp, eb):
        ev_pred = tf.expand_dims(tf.expand_dims(eb, axis=[1]), axis=[1])
        x = alg.mu_law_forward(inp, mu=self.mu_value)
        x = x * (2 ** (-1.0 * ev_pred))
        x = alg.mu_law_inverse(x, mu=self.mu_value)
        if self.norm_enable:
            x = x / tf.reduce_max(x)
        else:
            x = tf.clip_by_value(x, 0, 1)
        return x


class fusenet(Model):
    def __init__(self, mdl, mdl_dir, mdl_name):
        super().__init__(mdl, mdl_dir, mdl_name)
        self.fuse_gamma = 1.2
        self.low_median = 0.35
        self.high_median = 0.25
        self.adaptive_fuse = False

    def __call__(self, inputs, is_training=True):
        stack = self._fuse_images(inputs)
        stack = [inputs] + list(stack)
        self.out_layers, self.infos = self.mdl(stack, self.name, is_training)
        self.training = is_training
        self.out = self.out_layers[1]
        return self.out_layers[1]

    def _fuse_images(self, input):
        if self.adaptive_fuse:
            return mef.get_enhanced_priors_v2(input, [1.1, 1.3, 1.5], self.low_median, self.high_median)
        else:
            return mef.get_enhanced_priors_v1(input, self.fuse_gamma)

