import matplotlib
import platform
if platform.system() == "Darwin":
    matplotlib.use('TkAgg')
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import tensorflow as tf
import random
from tensorflow.contrib import layers
import sys
import os

def sample_z_fn(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


class VanillaGAN(object):
    def __init__(self, sess, num_epoch, batch_size, dataset, input_dim, z_dim):
        self.sess = sess
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.dataset = dataset
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.d_steps = 1
        self.g_steps = 1

    def build_net(self):
        # real input
        self.x = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='real_input')

        # noise sample z
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        # generator to generate fake data
        g = self._generator(self.z, reuse=False)

        # discriminator to judge fake data
        d_fake = self._discriminator(g, reuse=False)
        d_real = self._discriminator(self.x, reuse=True)

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

        """ Training """
        # optimizers
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

        self.d_optimizer = tf.train.AdamOptimizer().minimize(self.d_loss, var_list=d_vars)
        self.g_optimizer = tf.train.AdamOptimizer().minimize(self.g_loss, var_list=g_vars)

        """" Testing """
        self.sample_x = self._generator(self.z, reuse=True)

    def _generator(self, z, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):
            if self.dataset == 'mnist':
                net = layers.fully_connected(z, 128)
                net = layers.fully_connected(net, self.input_dim, activation_fn=None)
                net = tf.nn.sigmoid(net)

        return net

    def _discriminator(self, x, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            if self.dataset == 'mnist':
                net = layers.fully_connected(x, 128)
                net = layers.fully_connected(net, 1, activation_fn=None)

        return net

    def train(self, inputs, num_sample):
        # initialize all variables
        tf.global_variables_initializer().run()

        for epoch in range(self.num_epoch):
            G_loss = 0.0
            D_loss = 0.0

            # update D network
            for _ in range(self.d_steps):
                # sample real data
                X, _ = inputs.next_batch(self.batch_size)

                # sample random noises
                z_sample = sample_z_fn(self.batch_size, self.z_dim)

                _, d_loss = self.sess.run([self.d_optimizer, self.d_loss], feed_dict={self.x: X, self.z: z_sample})
                D_loss += d_loss

            D_loss /= self.d_steps

            # update G network
            for _ in range(self.g_steps):
                # sample random noises
                z_sample = sample_z_fn(self.batch_size, self.z_dim)

                _, g_loss = self.sess.run([self.g_optimizer, self.g_loss],
                                          feed_dict={self.x: X, self.z: z_sample})
                G_loss += g_loss
            G_loss /= self.g_steps

            # sample data for testing
            if epoch % 2000 == 0:
                z_sample = sample_z_fn(16, self.z_dim)
                self.visualize(epoch + 1, z_sample)

            # display current loss
            print "Epoch: %08d, d_loss= %.4f, g_loss=%.4f" % (epoch + 1, D_loss, G_loss)
            sys.stdout.flush()

    def visualize(self, epoch, z_sample):
        samples = self.sess.run(self.sample_x, feed_dict={self.z: z_sample})
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        if not os.path.isdir('output'):
            os.makedirs('output')

        with PdfPages('output/epoch_{}.pdf'.format(str(epoch).zfill(6))) as pdf:
            pdf.savefig(fig)
        plt.close()
