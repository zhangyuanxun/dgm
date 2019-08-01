import numpy as np
import tensorflow as tf
import random
from tensorflow.contrib import layers

# Fix random seed.
np.random.seed(0)
random.seed(0)
tf.set_random_seed(0)


class GAN(object):
    def __init__(self, sess, epoch, batch_size, dataset, input_dim, z_dim):
        self.sess = sess
        self.epoch = epoch
        self.batch_size = batch_size
        self.dataset = dataset
        self.input_dim = input_dim
        self.z_dim = z_dim

    def _build_net(self):
        # real input
        self.x = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='real_input')

        # noise sample z
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        # generator to generate fake data
        g = self._generator(self.z)

        # discriminator to judge fake data
        d_fake = self._discriminator(g)
        d_real = self._discriminator(self.x)

        # loss for discriminator
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_real, labels=tf.ones_like(d_real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_fake, labels=tf.zeros_like(d_fake)))

        # discriminator loss
        self.d_loss = d_loss_real + d_loss_fake

        # generator loss
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_fake, labels=tf.ones_like(d_fake)))

        # optimizers
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

        self.d_optimizer = tf.train.AdamOptimizer().minimize(self.d_loss, var_list=d_vars)
        self.g_optimizer = tf.train.AdamOptimizer().minimize(self.g_loss, var_list=g_vars)

    def _generator(self, z, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):
            if self.datasert == 'mnist':
                net = layers.fully_connected(z, 128)
                net = layers.fully_connected(net, self.input_dim)
                net = tf.nn.sigmoid(net)

        return net

    def _discriminator(self, x, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            if self.dataset == 'mnist':
                net = layers.fully_connected(x, 128)
                net = layers.fully_connected(net, 1, activation_fn=None)

        return net

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()
