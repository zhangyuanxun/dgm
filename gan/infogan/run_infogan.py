import input_data
import tensorflow as tf
from infogan import InfoGAN

if __name__ == '__main__':
    # get input data
    mnist_data = input_data.load_mnist_dataset('../../dataset/mnist_data', one_hot=True)
    num_sample = mnist_data.train.num_examples
    dataset = 'mnist'
    if dataset == 'mnist':
        input_dim = 784

    # define latent dimension
    z_dim = 16
    c_discrete_dim = 10
    c_continuous_dim = 2

    num_epoch = 1000000
    batch_size = 32

    # Launch the session
    with tf.Session() as sess:
        gan = InfoGAN(sess, num_epoch=num_epoch, batch_size=batch_size,
                      dataset=dataset, input_dim=input_dim, z_dim=z_dim, c_discrete_dim=c_discrete_dim,
                      c_continuous_dim=c_continuous_dim)

        # build generative adversarial network
        gan.build_net()

        # train the model
        gan.train(mnist_data.train, num_sample)
