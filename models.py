from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as tflayers
import tensorflow.contrib.layers as layers
import tflib as tl
import numpy as np
from pdb import set_trace


conv = partial(slim.conv2d, activation_fn=None)
dconv = partial(slim.conv2d_transpose, activation_fn=None)
fc = partial(tl.flatten_fully_connected, activation_fn=None)
relu = tf.nn.relu
lrelu = tf.nn.leaky_relu
batch_norm = partial(slim.batch_norm, scale=True, updates_collections=None)
instance_norm = slim.instance_norm

MAX_DIM = 64 * 16


def Genc(x, dim=64, n_layers=5, is_training=True):
    bn = partial(batch_norm, is_training=is_training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu)

    with tf.variable_scope('Genc', reuse=tf.AUTO_REUSE):
        z = x
        zs = []
        for i in range(n_layers):
            d = min(dim * 2**i, MAX_DIM)
            z = conv_bn_lrelu(z, d, 4, 2)
            zs.append(z)
        return zs


def Gdec(zs, _a, dim=64, n_layers=5, shortcut_layers=1, inject_layers=0, is_training=True,colorChannel=3):
    bn = partial(batch_norm, is_training=is_training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu)

    shortcut_layers = min(shortcut_layers, n_layers - 1)
    inject_layers = min(inject_layers, n_layers - 1)

    def _concat(z, z_, _a):
        feats = [z]
        if z_ is not None:
            feats.append(z_)
        if _a is not None:
            _a = tf.reshape(_a, [-1, 1, 1, tl.shape(_a)[-1]])
            _a = tf.tile(_a, [1, tl.shape(z)[1], tl.shape(z)[2], 1])
            feats.append(_a)
            
        return tf.concat(feats, axis=3)

    with tf.variable_scope('Gdec', reuse=tf.AUTO_REUSE):
        z = _concat(zs[-1], None, _a)

        # setting new attr
        # x_axis = tl.shape(z)[1]
        # y_axis = tl.shape(z)[2]
        # ch_count = int(np.round(tl.shape(z)[3] / x_axis / y_axis))
        # z = tflayers.fully_connected(z, x_axis * y_axis * ch_count)
        # z = tf.reshape(z, [-1, x_axis, y_axis, x_axis * y_axis * ch_count])

        for i in range(n_layers):
            if i < n_layers - 1:
                d = min(dim * 2**(n_layers - 1 - i), MAX_DIM)
                z = dconv_bn_relu(z, d, 4, 2)
                if shortcut_layers > i:
                    z = _concat(z, zs[n_layers - 2 - i], None)
                if inject_layers > i:
                    z = _concat(z, None, _a)
            else:
                x = z = tf.nn.tanh(dconv(z, colorChannel, 4, 2))
        return x


def D(x, n_att, dim=64, fc_dim=MAX_DIM, n_layers=5):
    conv_in_lrelu = partial(conv, normalizer_fn=instance_norm, activation_fn=lrelu)

    with tf.variable_scope('D', reuse=tf.AUTO_REUSE):
        y = x
        for i in range(n_layers):
            d = min(dim * 2**i, MAX_DIM)
            y = conv_in_lrelu(y, d, 4, 2)

        logit_gan = lrelu(fc(y, fc_dim))
        logit_gan = fc(logit_gan, 1)

        logit_att = lrelu(fc(y, fc_dim))
        logit_att = fc(logit_att, n_att)

        return logit_gan, logit_att


def D_info(x, n_att, dim=64, fc_dim=MAX_DIM, n_layers=5):
    conv_in_lrelu = partial(conv, normalizer_fn=instance_norm, activation_fn=lrelu)

    with tf.variable_scope('D', reuse=tf.AUTO_REUSE):
        y = x
        for i in range(n_layers):
            d = min(dim * 2**i, MAX_DIM)
            y = conv_in_lrelu(y, d, 4, 2)

        logit_gan = lrelu(fc(y, fc_dim))
        logit_gan = fc(logit_gan, 1)

        logit_att = lrelu(fc(y, fc_dim))
        logit_att = fc(logit_att, n_att)

        d2_flatten = layers.flatten(y)
        d3 = layers.fully_connected(d2_flatten, 1024, normalizer_fn=layers.batch_norm)
        r1 = layers.fully_connected(d3, 128, normalizer_fn=layers.batch_norm)
        r_cont_mu = layers.fully_connected(r1, 1, activation_fn=None)
        if False:
            r_cont_var = 1
        else:
            r_cont_logvar = layers.fully_connected(r1, 1, activation_fn=None)
            r_cont_var = tf.exp(r_cont_logvar)

        return logit_gan, logit_att, r_cont_mu, r_cont_var


def gradient_penalty(f, real, fake=None):
    def _interpolate(a, b=None):
        with tf.name_scope('interpolate'):
            if b is None:
                beta = tf.random_uniform(shape=tf.shape(a), minval=0., maxval=1.)
                dim = [x for x in range(a.shape.ndims)]
                _, variance = tf.nn.moments(a, dim)
                b = a + 0.5 * tf.sqrt(variance) * beta
            shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
            alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.get_shape().as_list())
            return inter

    with tf.name_scope('gradient_penalty'):
        x = _interpolate(real, fake)
        pred = f(x)
        if isinstance(pred, tuple):
            pred = pred[0]
        grad = tf.gradients(pred, x)[0]
        norm = tf.norm(slim.flatten(grad), axis=1)
        gp = tf.reduce_mean((norm - 1.)**2)
        return gp


def gradient_penalty_custom(f, real, fake=None):
    def _interpolate(a, b=None):
        with tf.name_scope('interpolate'):
            if b is None:
                beta = tf.random_uniform(shape=tf.shape(a), minval=0., maxval=1.)
                dim = [x for x in range(a.shape.ndims)]
                _, variance = tf.nn.moments(a, dim)
                b = a + 0.5 * tf.sqrt(variance) * beta
            shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
            alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.get_shape().as_list())
            return inter

    with tf.name_scope('gradient_penalty'):
        x = _interpolate(real, fake)
        x = tf.image.resize_images(x, [32, 32])
        pred = f(x)
        if isinstance(pred, tuple):
            pred = pred[4]
        grad = tf.gradients(pred, x)[0]
        norm = tf.norm(slim.flatten(grad), axis=1)
        gp = tf.reduce_mean((norm - 1.)**2)
        return gp
