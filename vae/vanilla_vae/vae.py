import numpy as np
import tensorflow as tf


def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


class VAE(object):
    """Variation Autoencoder (VAE) implementation using TensorFlow.

    Reference "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """
    def __init__(self, network_architecture, batch_size, learning_rate=0.001):
        self.network_architecture = network_architecture
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # tf input data
        self.X = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])

        # create variational autoencoder model
        self._create_network()

        # create loss and optimizer
        self._create_loss_optimizer()

        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)

    def _initialize_weights(self):
        """
        Initialize network weights
        :return:
        :rtype:
        """
        n_hidden_encoder_1 = self.network_architecture["n_hidden_encoder_1"]
        n_hidden_encoder_2 = self.network_architecture["n_hidden_encoder_2"]
        n_hidden_decoder_1 = self.network_architecture["n_hidden_decoder_1"]
        n_hidden_decoder_2 = self.network_architecture["n_hidden_decoder_2"]
        n_input = self.network_architecture["n_input"]
        n_z = self.network_architecture["n_z"]

        weights = dict()

        weights['encoder_weights'] = {
            'h1': tf.Variable(xavier_init(n_input, n_hidden_encoder_1)),
            'h2': tf.Variable(xavier_init(n_hidden_encoder_1, n_hidden_encoder_2)),
            'z_mean': tf.Variable(xavier_init(n_hidden_encoder_2, n_z)),
            'z_var': tf.Variable(xavier_init(n_hidden_encoder_2, n_z))
        }
        weights['encoder_biases'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_encoder_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_encoder_2], dtype=tf.float32)),
            'z_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'z_var': tf.Variable(tf.zeros([n_z], dtype=tf.float32))
        }
        weights['decoder_weights'] = {
            'h1': tf.Variable(xavier_init(n_z, n_hidden_decoder_1)),
            'h2': tf.Variable(xavier_init(n_hidden_decoder_1, n_hidden_decoder_2)),
            'z_mean':tf.Variable(xavier_init(n_hidden_decoder_2, n_input)),
            'z_var': tf.Variable(xavier_init(n_hidden_decoder_2, n_input))
        }
        weights['decoder_biases'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_decoder_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_decoder_2], dtype=tf.float32)),
            'z_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
            'z_var': tf.Variable(tf.zeros([n_input], dtype=tf.float32))
        }
        return weights

    def _create_network(self):
        # Initialize autoencode network weights and biases
        weights = self._initialize_weights()

        # build recognition network (encoder network)
        # Use recognition network to determine mean and (log) variance of Gaussian distribution in latent space
        self.z_mean, self.z_log_var = self._encoder_network(weights['encoder_weights'], weights['encoder_biases'])

        # sample noise
        epsilon = tf.random_normal(shape=tf.shape(self.z_mean), mean=0, stddev=1, dtype=tf.float32)

        # z = mu + sigma * epsilon to approximate from Gaussian distribution z ~ q(z | x)
        self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_var)), epsilon))

        # build generator network (decoder network)
        self.X_hat = self._decoder_network(weights['decoder_weights'], weights['decoder_biases'])

    def _encoder_network(self, weights, biases):
        """
            Generate encoder network (recognition network), which maps inputs X onto a normal
            distribution in latent space. The transformation is parametrized and can be learned.
        :param weights:
        :type latent_dim:
        :return:
        :rtype:
        """
        # create first hidden encoder layer
        encoder_layer1 = tf.nn.softplus(tf.matmul(self.X, weights['h1']) + biases['b1'])

        # creat second hidden encoder layer
        encoder_layer2 = tf.nn.softplus(tf.matmul(encoder_layer1, weights['h2']) + biases['b2'])

        # create latent layer z mean encoder
        z_mean = tf.add(tf.matmul(encoder_layer2, weights['z_mean']), biases['z_mean'])

        # create latent layer z log of variance
        z_logvar = tf.add(tf.matmul(encoder_layer2, weights['z_var']), biases['z_var'])

        return z_mean, z_logvar

    def _decoder_network(self, weights, biases):
        """ Generate probabilistic decoder (generator network), which maps points in latent space
            onto a Bernoulli distribution in data space. The transformation is parametrized and can be learned.
        :param latent_dim:
        :type latent_dim:
        :return:
        :rtype:
        """
        # create first decoder layer
        decoder_layer1 = tf.nn.softplus(tf.matmul(self.z, weights['h1']) + biases['b1'])

        # create second decoder layer
        decoder_layer2 = tf.nn.softplus(tf.matmul(decoder_layer1, weights['h2']) + biases['b2'])

        # reconstruct x
        X_hat = tf.nn.sigmoid(tf.matmul(decoder_layer2, weights['z_mean']) + biases['z_mean'])

        return X_hat

    def _create_loss_optimizer(self):
        """
        The loss of vae includes two parts: reconstruction loss and KL loss
        :return:
        :rtype:
        """
        # (1) reconstruction loss
        reconstruction_loss = -tf.reduce_sum(self.X * tf.log(tf.maximum(self.X_hat, 1e-10))
                                                  + (1 - self.X) * tf.log(tf.maximum(1 - self.X_hat, 1e-10)), 1)
        # (2) KL loss
        kl_loss = -0.5 * tf.reduce_sum(1 + self.z_log_var - tf.square(self.z_mean) - tf.exp(self.z_log_var), 1)

        # compute loss
        self.loss = tf.reduce_mean(reconstruction_loss + kl_loss)

        # add optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def fit(self, X):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        opt, loss = self.sess.run((self.optimizer, self.loss), feed_dict={self.X: X})
        return loss

    def embedding(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.X: X})

    def sampling(self, z_mean=None):
        """ Generate data by sampling from latent space.

        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent
        space.
        """
        if z_mean is None:
            z_mean = np.random.normal(size=self.network_architecture["n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.X_hat,
                             feed_dict={self.z: z_mean})

    def reconstruct(self, X):
        """
        Use VAE to reconstruct given data.
        :param X:
        :type X:
        :return:
        :rtype:
        """
        return self.sess.run(self.X_hat, feed_dict={self.X: X})
