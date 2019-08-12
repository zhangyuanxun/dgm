import input_data
import tensorflow as tf
from vanilla_gan import VanillaGAN

if __name__ == '__main__':
    # get input data
    mnist_data = input_data.load_mnist_dataset('../../dataset/mnist_data', one_hot=True)
    num_sample = mnist_data.train.num_examples
    dataset = 'mnist'
    if dataset == 'mnist':
        input_dim = 784

    # define latent dimension
    z_dim = 100

    num_epoch = 100000
    batch_size = 100

    # Launch the session
    with tf.Session() as sess:
        gan = VanillaGAN(sess, num_epoch=num_epoch, batch_size=batch_size,
                         dataset=dataset, input_dim=input_dim, z_dim=z_dim)

        # build generative adversarial network
        gan.build_net()

        # train the model
        gan.train(mnist_data.train, num_sample)
