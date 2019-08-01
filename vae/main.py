from vae import VAE
from train import*
import input_data
import matplotlib.pyplot as plt


def demo_reconstruction():
    # get input data
    mnist_data = input_data.read_data_sets('../dataset/mnist_data', one_hot=True)
    num_sample = mnist_data.train.num_examples
    batch_size = 100

    network_architecture = \
        dict(n_hidden_encoder_1=500,  # 1st layer encoder neurons
             n_hidden_encoder_2=500,  # 2nd layer encoder neurons
             n_hidden_decoder_1=500,  # 1st layer decoder neurons
             n_hidden_decoder_2=500,  # 2nd layer decoder neurons
             n_input=784,  # MNIST data input (img shape: 28*28)
             n_z=20)  # dimensionality of latent space

    # define model
    vae_model = VAE(network_architecture, batch_size=batch_size)

    # train the model
    train(model=vae_model, inputs=mnist_data.train, num_epoch=10, num_sample=num_sample, batch_size=batch_size)

    # test reconstruct
    X_test = mnist_data.test.next_batch(100)[0]
    X_hat = vae_model.reconstruct(X_test)
    plt.figure(figsize=(8, 12))
    for i in range(5):
        plt.subplot(5, 2, 2 * i + 1)
        plt.imshow(X_test[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Test input")
        plt.colorbar()
        plt.subplot(5, 2, 2 * i + 2)
        plt.imshow(X_hat[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Reconstruction")
        plt.colorbar()
    plt.tight_layout()
    plt.show()


def demo_embedding():
    # get input data
    mnist_data = input_data.read_data_sets('../dataset/mnist_data', one_hot=True)
    num_sample = mnist_data.train.num_examples
    batch_size = 100

    network_architecture = \
        dict(n_hidden_encoder_1=500,  # 1st layer encoder neurons
             n_hidden_encoder_2=500,  # 2nd layer encoder neurons
             n_hidden_decoder_1=500,  # 1st layer decoder neurons
             n_hidden_decoder_2=500,  # 2nd layer decoder neurons
             n_input=784,             # MNIST data input (img shape: 28*28)
             n_z=2)                   # dimensionality of latent space

    # define model
    vae_model = VAE(network_architecture, batch_size=batch_size)

    # train the model
    train(model=vae_model, inputs=mnist_data.train, num_epoch=50, num_sample=num_sample, batch_size=batch_size)

    # embedding data into 2D space
    X_sample, Y_sample = mnist_data.test.next_batch(5000)
    z_mu = vae_model.embedding(X_sample)

    plt.figure(figsize=(8, 6))
    plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.argmax(Y_sample, 1))
    plt.colorbar()
    plt.grid()
    plt.show()


def demo_sampling():
    # get input data
    mnist_data = input_data.read_data_sets('../dataset/mnist_data', one_hot=True)
    num_sample = mnist_data.train.num_examples
    batch_size = 100

    network_architecture = \
        dict(n_hidden_encoder_1=500,  # 1st layer encoder neurons
             n_hidden_encoder_2=500,  # 2nd layer encoder neurons
             n_hidden_decoder_1=500,  # 1st layer decoder neurons
             n_hidden_decoder_2=500,  # 2nd layer decoder neurons
             n_input=784,             # MNIST data input (img shape: 28*28)
             n_z=2)                   # dimensionality of latent space

    network_architecture = {}
    # define model
    vae_model = VAE(network_architecture, batch_size=batch_size)

    # train the model
    train(model=vae_model, inputs=mnist_data.train, num_epoch=10, num_sample=num_sample, batch_size=batch_size)

    nx = ny = 20
    x_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny)

    canvas = np.empty((28 * ny, 28 * nx))
    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
            z_mu = np.array([[xi, yi]] * vae_model.batch_size)
            x_mean = vae_model.sampling(z_mu)
            canvas[(nx - i - 1) * 28:(nx - i) * 28, j * 28:(j + 1) * 28] = x_mean[0].reshape(28, 28)

    plt.figure(figsize=(8, 10))
    Xi, Yi = np.meshgrid(x_values, y_values)
    plt.imshow(canvas, origin="upper", cmap="gray")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # get input data
    demo_reconstruction()