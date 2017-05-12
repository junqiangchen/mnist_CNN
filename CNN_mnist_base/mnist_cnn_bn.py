#
#  mnist_cnn_bn.py   date. 5/21/2016
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

import pandas as pd
from my_nn_lib import Convolution2D, MaxPooling2D
from my_nn_lib import FullConnected, ReadOutLayer

chkpt_file = 'F:\PycharmProject\CNN_mnist_base\model\mymodel'


def batch_norm(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


#

def training(loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op


def evaluation(y_pred, y):
    correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    return accuracy


def mlogloss(predicted, actual):
    '''
      args.
         predicted : predicted probability
                    (sum of predicted proba should be 1.0)
         actual    : actual value, label
    '''

    def inner_fn(item):
        eps = 1.e-15
        item1 = min(item, (1 - eps))
        item1 = max(item, eps)
        res = np.log(item1)

        return res

    nrow = actual.shape[0]
    ncol = actual.shape[1]

    mysum = sum([actual[i, j] * inner_fn(predicted[i, j])
                 for i in range(nrow) for j in range(ncol)])

    ans = -1 * mysum / nrow

    return ans


#

# Create the model
def inference(x, y_, keep_prob, phase_train):
    with tf.variable_scope('conv_1'):
        conv1 = Convolution2D(x, (28, 28), 1, 32, (5, 5), activation='none')
        conv1_bn = batch_norm(conv1.output(), 32, phase_train)
        conv1_out = tf.nn.relu(conv1_bn)

        pool1 = MaxPooling2D(conv1_out)
        pool1_out = pool1.output()

    with tf.variable_scope('conv_2'):
        conv2 = Convolution2D(pool1_out, (14, 14), 32, 64, (5, 5),
                              activation='none')
        conv2_bn = batch_norm(conv2.output(), 64, phase_train)
        conv2_out = tf.nn.relu(conv2_bn)

        pool2 = MaxPooling2D(conv2_out)
        pool2_out = pool2.output()

        pool2_flat = tf.reshape(pool2_out, [-1, 7 * 7 * 64])

    with tf.variable_scope('fc1'):
        fc1 = FullConnected(pool2_flat, 7 * 7 * 64, 1024)
        fc1_out = fc1.output()
        fc1_dropped = tf.nn.dropout(fc1_out, keep_prob)

    y_pred = ReadOutLayer(fc1_dropped, 1024, 10).output()

    loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_pred), reduction_indices=[1]))
    accuracy = evaluation(y_pred, y_)

    return loss, accuracy, y_pred


# Convert class labels from scalars to one-hot vectors
# 0 => [1 0 0 0 0 0 0 0 0 0]
# 1 => [0 1 0 0 0 0 0 0 0 0]
# 2 => [0 0 1 0 0 0 0 0 0 0]
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


# Serve data by batches
def next_batch(batch_size):
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]


#
if __name__ == '__main__':
    TASK = 'train'  # 'train' or 'test'
    VALIDATION_SIZE = 1000
    # Read MNIST data set (Train data from CSV file)
    csvdata = pd.read_csv('train.csv')
    data = csvdata.iloc[:, :].values
    np.random.shuffle(data)
    # Extracting images and labels from given data
    # For images
    images = data[:, 1:]
    images = images.astype(np.float)

    # Normalize from [0:255] => [0.0:1.0]
    images = np.multiply(images, 1.0 / 255.0)
    # For labels
    labels_flat = data[:, 0]
    labels_count = np.unique(labels_flat).shape[0]
    labels = dense_to_one_hot(labels_flat, labels_count)
    labels = labels.astype(np.uint8)

    # Split data into training & validation
    validation_images = images[:VALIDATION_SIZE]
    validation_labels = labels[:VALIDATION_SIZE]

    train_images = images[VALIDATION_SIZE:]
    train_labels = labels[VALIDATION_SIZE:]
    # Variables
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
    phase_train = tf.placeholder(tf.bool, name='phase_train')

    loss, accuracy, y_pred = inference(x, y_,
                                       keep_prob, phase_train)

    # Train
    lr = 0.01
    train_step = tf.train.AdagradOptimizer(lr).minimize(loss)
    vars_to_train = tf.trainable_variables()  # option-1
    vars_for_bn1 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='conv_1/bn')
    vars_for_bn2 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='conv_2/bn')
    vars_to_train = list(set(vars_to_train).union(set(vars_for_bn1)))
    vars_to_train = list(set(vars_to_train).union(set(vars_for_bn2)))

    if TASK == 'test' or os.path.exists(chkpt_file):
        restore_call = True
        vars_all = tf.all_variables()
        vars_to_init = list(set(vars_all) - set(vars_to_train))
        init = tf.initialize_variables(vars_to_init)
    elif TASK == 'train':
        restore_call = False
        init = tf.initialize_all_variables()
    else:
        print('Check task switch.')

    saver = tf.train.Saver(vars_to_train)  # option-1
    # saver = tf.train.Saver()                   # option-2

    epochs_completed = 0
    index_in_epoch = 0
    num_examples = train_images.shape[0]
    with tf.Session() as sess:
        # if TASK == 'train':              # add in option-2 case
        sess.run(init)  # option-1

        if restore_call:
            # Restore variables from disk.
            saver.restore(sess, chkpt_file)

        if TASK == 'train':
            print('\n Training...')
            for i in range(100):
                batch_xs, batch_ys = next_batch(200)
                train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.8,
                                phase_train: True})
                if i % 10 == 0:
                    cv_fd = {x: batch_xs, y_: batch_ys, keep_prob: 1.0,
                             phase_train: False}
                    train_loss = loss.eval(cv_fd)
                    train_accuracy = accuracy.eval(cv_fd)

                    print('  step, loss, accurary = %6d: %8.4f, %8.4f' % (i,
                                                                          train_loss, train_accuracy))

        # Test trained model
        test_fd = {x: validation_images, y_: validation_labels,
                   keep_prob: 1.0, phase_train: False}
        print(' accuracy = %8.4f' % accuracy.eval(test_fd))
        # Multiclass Log Loss
        pred = y_pred.eval(test_fd)
        act = validation_labels
        print(' multiclass logloss = %8.4f' % mlogloss(pred, act))

        # Save the variables to disk.
        if TASK == 'train':
            save_path = saver.save(sess, chkpt_file)
            print("Model saved in file: %s" % save_path)
