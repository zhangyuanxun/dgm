import numpy as np
import tensorflow as tf


def train(model, inputs, num_epoch, num_sample, batch_size):
    """
    Implement the train function
    """
    for epoch in range(num_epoch):
        train_epoch(model, inputs, epoch, num_sample, batch_size)


def train_epoch(model, inputs, epoch, num_sample, batch_size):
    """
    Implement the logic of each epoch
    """

    # Compute number of batches in one epoch (one full pass over the training set)
    num_batch = (num_sample + batch_size - 1) // batch_size
    losses = []
    for i in range(num_batch):
        X, _ = inputs.next_batch(batch_size)
        loss = model.fit(X)
        losses.append(loss)

    # display current cost
    print "Epoch: %04d, loss= %.4f" % (epoch + 1, sum(losses) / len(losses))
