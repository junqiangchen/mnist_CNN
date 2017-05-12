'''
2-D Convolutional Neural Networks using TensorFlow library for jieheganjun detection.
Author: junqiangChen

Train instances: 50000 number images with vector format (1 number = 1 x 1024)
Test instances: 10000 number images with vector format  (1 number = 1 x 1024)
'''

from __future__ import division
import numpy as np
import pandas as pd
import sys
import tensorflow as tf
from PIL import Image, ImageFilter

# Parameters
DROPOUT_CONV = 0.8
DROPOUT_HIDDEN = 0.7
image_size = 1024
image_width = image_height = 32
image_labels = 2


# Weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# Weight initialization (Xavier's init)
def weight_xavier_init(shape, n_inputs, n_outputs, uniform=True):
    if uniform:
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        initial = tf.random_uniform(shape, -init_range, init_range)
        return tf.Variable(initial)
    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        initial = tf.truncated_normal(shape, stddev=stddev)
        return tf.Variable(initial)


# Bias initialization
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def batchnorm(Ylogits, n_out, is_test, convolutional=True):
    """
        Batch normalization on convolutional maps.
        Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
        Args:
            Ylogits:           Tensor, 4D BHWD input maps
            n_out:       integer, depth of input maps
            is_test: boolean tf.Varialbe, true indicates training phase
            scope:       string, variable scope
        Return:
            normed:      batch-normalized maps
        """
    # adding the iteration prevents from averaging across non-existing iterations
    beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                       name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                        name='gamma', trainable=True)
    exp_moving_avg = tf.train.ExponentialMovingAverage(decay=0.9)
    bnepsilon = 0.001
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, beta, gamma, bnepsilon)
    return Ybn


# 2D convolution
def conv2d(X, W):
    return tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')


# Max Pooling
def max_pool_2x2(X):
    return tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


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


# Create CNN Model
X = tf.placeholder('float', shape=[None, image_size])
Y_gt = tf.placeholder('float', shape=[None, image_labels])
drop_conv = tf.placeholder('float')
drop_hidden = tf.placeholder('float')
# Model Parameters
# CNN model
X1 = tf.reshape(X, [-1, image_width, image_height, 1])  # shape=(?, 32, 32, 1)
# Layer 1
W1 = weight_xavier_init(shape=[3, 3, 1, 32], n_inputs=3 * 3 * 1,
                        n_outputs=32)  # 3x3x1 conv, 32 outputs,image shape[32,32]->[32,32]
B1 = bias_variable([32])

conv1 = conv2d(X1, W1) + B1
l1_conv = tf.nn.relu(conv1)  # shape=(?, 32, 32, 32)
l1_drop = tf.nn.dropout(l1_conv, drop_conv)
# Layer 2
W2 = weight_xavier_init(shape=[3, 3, 32, 32], n_inputs=3 * 3 * 32,
                        n_outputs=32)  # 3x3x1 conv, 32 outputs,image shape[32,32]->[16,16]
B2 = bias_variable([32])

conv2 = conv2d(l1_drop, W2) + B2
l2_conv = tf.nn.relu(conv2)  # shape=(?, 32, 32, 32)
l2_pool = max_pool_2x2(l2_conv)  # shape=(?, 16, 16, 32)
l2_drop = tf.nn.dropout(l2_pool, drop_conv)
# Layer 3
W3 = weight_xavier_init(shape=[3, 3, 32, 64], n_inputs=3 * 3 * 32,
                        n_outputs=64)  # 3x3x32 conv, 64 outputs,image shape[16,16]->[16,16]
B3 = bias_variable([64])

conv3 = conv2d(l2_drop, W3) + B3
l3_conv = tf.nn.relu(conv3)  # shape=(?, 16, 16, 64)
l3_drop = tf.nn.dropout(l3_conv, drop_conv)
# Layer 4
W4 = weight_xavier_init(shape=[3, 3, 64, 64],
                        n_inputs=3 * 3 * 64,
                        n_outputs=64)  # 3x3x64 conv, 64 outputs,image shape[16,16]->[8,8]
B4 = bias_variable([64])

conv4 = conv2d(l3_drop, W4) + B4
l4_conv = tf.nn.relu(conv4)  # shape=(?, 16, 16, 64)
l4_pool = max_pool_2x2(l4_conv)  # shape=(?, 8, 8, 64)
l4_drop = tf.nn.dropout(l4_pool, drop_conv)

# Layer 5 - FC1
W5_FC1 = weight_xavier_init(shape=[64 * 8 * 8, 512],
                            n_inputs=64 * 8 * 8, n_outputs=512)  # FC: 64x8x8 inputs, 512 outputs
B5_FC1 = bias_variable([512])

l5_flat = tf.reshape(l4_drop, [-1, W5_FC1.get_shape().as_list()[0]])  # shape=(?, 512)
FC1 = tf.matmul(l5_flat, W5_FC1) + B5_FC1
l5_feed = tf.nn.relu(FC1)
l5_drop = tf.nn.dropout(l5_feed, drop_hidden)
# Layer 6 - FC2
W6_FC2 = weight_xavier_init(shape=[512, image_labels],
                            n_inputs=512, n_outputs=image_labels)  # FC: 512 inputs, 2 outputs (labels)
B6_FC2 = bias_variable([image_labels])

Y_pred = tf.nn.softmax(tf.matmul(l5_drop, W6_FC2) + B6_FC2)  # shape=(?, 2)
predict = tf.argmax(Y_pred, 1)
'''
TensorFlow Session
'''
# start TensorFlow session
init = tf.initialize_all_variables()
saver = tf.train.Saver()
sess = tf.InteractiveSession()
sess.run(init)
saver.restore(sess, 'F:\PycharmProject\CNN_jieheganjun2017.3.8_NoBN\model\mymodel')


def predictint(test_images):
    predictvalue = np.zeros(test_images.shape[0])
    for i in range(0, test_images.shape[0]):
        imvalue = test_images[i]
        predictvalue[i] = predict.eval(
            feed_dict={X: [imvalue], drop_conv: DROPOUT_CONV, drop_hidden: DROPOUT_HIDDEN},
            session=sess)

    return predictvalue


def imageprepare(imagepath):
    im = Image.open(imagepath)
    im_array = np.array(im)
    test_images = im_array.reshape((1, -1))
    test_images = test_images.astype(np.float)
    # convert from [0:255] => [0.0:1.0]
    test_images = np.multiply(test_images, 1.0 / 255.0)
    print('test_images({0[0]},{0[1]})'.format(test_images.shape))
    return test_images


def imagecsvprepare(csvPath):
    test_imagesdata = pd.read_csv(csvPath)
    test_images = test_imagesdata.iloc[:, 1:].values
    test_labels = test_imagesdata[[0]].values
    test_images = test_images.astype(np.float)
    # convert from [0:255] => [0.0:1.0]
    test_images = np.multiply(test_images, 1.0 / 255.0)
    print('test_images({0[0]},{0[1]})'.format(test_images.shape))
    return test_images, test_labels


def main(argv):
    """
    Main function.
    """
    if (argv == 1):
        imvalue = imageprepare("19.jpg")
        predint = predictint(imvalue)
        print("predictvalue %d" % (predint[0]))  # first value in list
    if (argv == 2):
        test_images, test_labels = imagecsvprepare('test.csv')
        predint = predictint(test_images)
        Pcount = 0
        Ncount = 0
        PNcount = 0
        NPcount = 0
        for i in range(predint.shape[0]):
            if test_labels[i] == predint[i] == 1:
                Pcount += 1
            if test_labels[i] == predint[i] == 0:
                Ncount += 1
            if test_labels[i] == 1 and predint[i] == 0:
                NPcount += 1
            if test_labels[i] == 0 and predint[i] == 1:
                PNcount += 1
        print("Positive: %d,Negative:%d,PN:%d,NP:%d" % (Pcount, Ncount, PNcount, NPcount))  # first value in list


if __name__ == "__main__":
    main(2)
