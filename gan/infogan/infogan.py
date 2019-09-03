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


continuous_mean = 0.0
continuous_std = 1.0


def sample_z_fn(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def sample_c_discrete_fn(m, n):
    return np.random.multinomial(1, n * [0.1], size=m)


def sample_c_continuous_fn(m, n):
    return np.random.normal(loc=continuous_mean, scale=continuous_std, size=[m, n])


def log_prob_gaussian(x_val):
    epsilon = (x_val - continuous_mean) / (continuous_std + 1e-8)
    log_prob = - 0.5 * np.log(2 * np.pi) - tf.log(continuous_std + 1e-8) - 0.5 * tf.square(epsilon)
    return log_prob


class InfoGAN(object):
    def __init__(self, sess, num_epoch, batch_size, dataset, input_dim, z_dim, c_discrete_dim, c_continuous_dim):
        self.sess = sess
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.dataset = dataset
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.c_discrete_dim = c_discrete_dim
        self.c_continuous_dim = c_continuous_dim
        self.coeff = 1.0

    def build_net(self):
        # real input
        self.x = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='real_input')

        # noise sample z
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        # structured latent sample discrete c
        self.c_discrete = tf.placeholder(tf.float32, [None, self.c_discrete_dim], name='c_discrete')

        # structured latent sample continuous c
        self.c_continuous = tf.placeholder(tf.float32, [None, self.c_continuous_dim], name='c_continuous')

        # generator to generate fake data
        g = self._generator(self.z, self.c_discrete, self.c_continuous, reuse=False)

        # discriminator to judge fake data
        d_fake = self._discriminator(g, reuse=False)
        d_real = self._discriminator(self.x, reuse=True)

        # output of auxiliary network Q for discrete value
        q_discrete = self._Q_discrete(g)

        # output of auxiliary network Q for continuous value
        q_continuous = self._Q_continuous(g)

        # discriminator loss
        self.d_loss = -tf.reduce_mean(tf.log(d_real + 1e-8) + tf.log(1 - d_fake + 1e-8))

        # generator loss
        self.g_loss = -tf.reduce_mean(tf.log(d_fake + 1e-8))

        # auxiliary discrete loss
        self.q_discrete_loss = -tf.reduce_mean(tf.reduce_sum(tf.log(q_discrete + 1e-8) * self.c_discrete, 1))

        # auxiliary continuous loss
        self.q_continuous_loss = -self.coeff * tf.reduce_mean(tf.reduce_sum(q_continuous * tf.exp(log_prob_gaussian(self.c_continuous)), 1))

        """ Training """
        # optimizers
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
        q_discrete_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'q_discrete')
        q_continuous_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'q_continuous')

        self.d_optimizer = tf.train.AdamOptimizer().minimize(self.d_loss, var_list=d_vars)
        self.g_optimizer = tf.train.AdamOptimizer().minimize(self.g_loss, var_list=g_vars)
        self.q_discrete_optimizer = tf.train.AdamOptimizer().minimize(self.q_discrete_loss, var_list=g_vars + q_discrete_vars)
        self.q_continuous_optimizer = tf.train.AdamOptimizer().minimize(self.q_continuous_loss, var_list=g_vars + q_continuous_vars)

        """" Testing """
        self.sample_x = self._generator(self.z, self.c_discrete, self.c_continuous, reuse=True)

    def _generator(self, z, c_discrete, c_continuous, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):
            if self.dataset == 'mnist':
                inputs = tf.concat(axis=1, values=[z, c_discrete, c_continuous])
                net = layers.fully_connected(inputs, 256)
                prob = layers.fully_connected(net, self.input_dim, activation_fn=tf.nn.sigmoid)
        return prob

    def _discriminator(self, x, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            if self.dataset == 'mnist':
                net = layers.fully_connected(x, 128)
                prob = layers.fully_connected(net, 1, activation_fn=tf.nn.sigmoid)
        return prob

    # auxiliary network Q for discrete latent variable
    def _Q_discrete(self, x, reuse=False):
        with tf.variable_scope("q_discrete", reuse=reuse):
            net = layers.fully_connected(x, 128)
            prob = layers.fully_connected(net, self.c_discrete_dim, activation_fn=tf.nn.softmax)

        return prob

    def _Q_continuous(self, x, reuse=False):
        with tf.variable_scope("q_continuous", reuse=reuse):
            net = layers.fully_connected(x, 128)
            net = layers.fully_connected(net, self.c_continuous_dim, activation_fn=None)
            log_prob = log_prob_gaussian(net)

        return log_prob

    def train(self, inputs, num_sample):
        # initialize all variables
        tf.global_variables_initializer().run()

        for epoch in range(self.num_epoch):
            # sample real data
            X, _ = inputs.next_batch(self.batch_size)

            # sample random noises
            z_noise = sample_z_fn(self.batch_size, self.z_dim)

            # sample discrete noises
            c_discrete_noise = sample_c_discrete_fn(self.batch_size, self.c_discrete_dim)

            # sample continuous noises
            c_continuous_noise = sample_c_continuous_fn(self.batch_size, self.c_continuous_dim)

            # update D network
            _, D_loss = self.sess.run([self.d_optimizer, self.d_loss], feed_dict={self.x: X, self.z: z_noise,
                                                                                  self.c_discrete: c_discrete_noise,
                                                                                  self.c_continuous: c_continuous_noise})
            # update G network
            _, G_loss = self.sess.run([self.g_optimizer, self.g_loss],
                                      feed_dict={self.z: z_noise, self.c_discrete: c_discrete_noise,
                                                 self.c_continuous: c_continuous_noise})

            # update Q network for discrete
            _, Q_discrete_loss = self.sess.run([self.q_discrete_optimizer, self.q_discrete_loss],
                                      feed_dict={self.z: z_noise, self.c_discrete: c_discrete_noise,
                                                 self.c_continuous: c_continuous_noise})

            # update Q network for discrete
            _, Q_continuous_loss = self.sess.run([self.q_continuous_optimizer, self.q_continuous_loss],
                                      feed_dict={self.z: z_noise, self.c_discrete: c_discrete_noise,
                                                 self.c_continuous: c_continuous_noise})

            # sample data for testing
            if epoch % 1000 == 0:
                z_noise = sample_z_fn(16, self.z_dim)
                idx = np.random.randint(0, 10)

                c_discrete_noise = np.zeros([16, self.c_discrete_dim])
                c_discrete_noise[range(16), idx] = 1

                c_continuous_noise = np.zeros([16, self.c_continuous_dim])

                # adjust continuous noise (-2, 2), keeping a dimensional fixed
                i = np.random.randint(0, 2)
                base = -2.0
                for j in range(16):
                    c_continuous_noise[j, i] = base
                    base += 0.26

                self.visualize(epoch + 1, z_noise, c_discrete_noise, c_continuous_noise, lable=i)

                # display current loss
                print "Epoch: %08d, d_loss= %.4f, g_loss=%.4f, q_discrete_loss=%.4f, q_continuous_loss=%.4f" \
                      % (epoch + 1, D_loss, G_loss, Q_discrete_loss, Q_continuous_loss)
                sys.stdout.flush()

    def visualize(self, epoch, z_noise, c_discrete_noise, c_continuous_noise, lable):
        samples = self.sess.run(self.sample_x, feed_dict={self.z: z_noise,
                                                          self.c_discrete: c_discrete_noise,
                                                          self.c_continuous: c_continuous_noise})
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
        file_name = str(epoch) + '_' + str(lable)
        with PdfPages('output/epoch_{}.pdf'.format(str(file_name).zfill(8))) as pdf:
            pdf.savefig(fig)
        plt.close(fig)
