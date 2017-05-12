'''
2-D Convolutional Neural Networks using TensorFlow library for Kaggle competition.

Target competition on Kaggle: https://www.kaggle.com/c/digit-recognizer
Author: Taegyun Jeon
Project: https://github.com/tgjeon/cnnForMnist

Train instances: 42000 number images with vector format (1 number = 1 x 748)
Test instances: 20000 number images with vector format  (1 number = 1 x 748)
'''

import numpy as np
import pandas as pd
import sys
import tensorflow as tf
from PIL import Image, ImageFilter

# Parameters
DROPOUT_CONV = 0.8
DROPOUT_HIDDEN = 0.6

image_size = 784
image_width = image_height = 28
image_labels = 10


# Weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# Weight initialization (Xavier's init)
def weight_xavier_init(n_inputs, n_outputs, uniform=True):
    if uniform:
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)


# Bias initialization
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


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


def imageprepare(imagepath):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(imagepath).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheigth = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # caculate horizontal pozition
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    # newImage.save("sample.png")

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    return tva
    # print(tva)


def imagecsvprepare(csvPath):
    test_imagesdata = pd.read_csv(csvPath)
    test_images = test_imagesdata.iloc[:, 1:].values
    test_labels = test_imagesdata[[0]].values
    test_images = test_images.astype(np.float)
    # convert from [0:255] => [0.0:1.0]
    test_images = np.multiply(test_images, 1.0 / 255.0)
    print('test_images({0[0]},{0[1]})'.format(test_images.shape))
    return test_images, test_labels


def predictint(test_images):
    '''
    Create model with 2D CNN
    '''
    # Create Input and Output
    X = tf.placeholder('float', shape=[None, image_size])  # mnist data image of shape 28*28=784
    Y_gt = tf.placeholder('float', shape=[None, image_labels])  # 0-9 digits recognition => 10 classes
    drop_conv = tf.placeholder('float')
    drop_hidden = tf.placeholder('float')

    # Model Parameters
    W1 = tf.get_variable("W1", shape=[5, 5, 1, 32], initializer=weight_xavier_init(5 * 5 * 1,
                                                                                   32))  # 5x5x1 conv, 32 outputs,image shape[28,28]->[14,14]
    W2 = tf.get_variable("W2", shape=[5, 5, 32, 64], initializer=weight_xavier_init(5 * 5 * 32,
                                                                                    64))  # 5x5x32 conv, 64 outputs,image shape[14,14]->[7,7]
    W3_FC1 = tf.get_variable("W3_FC1", shape=[64 * 7 * 7, 1024],
                             initializer=weight_xavier_init(64 * 7 * 7, 1024))  # FC: 64x7x7 inputs, 1024 outputs
    W4_FC2 = tf.get_variable("W4_FC2", shape=[1024, image_labels],
                             initializer=weight_xavier_init(1024, image_labels))  # FC: 1024 inputs, 10 outputs (labels)

    # W1 = weight_variable([5, 5, 1, 32])              # 5x5x1 conv, 32 outputs
    # W2 = weight_variable([5, 5, 32, 64])             # 5x5x32 conv, 64 outputs
    # W3_FC1 = weight_variable([64 * 7 * 7, 1024])     # FC: 64x7x7 inputs, 1024 outputs
    # W4_FC2 = weight_variable([1024, labels_count])   # FC: 1024 inputs, 10 outputs (labels)

    B1 = bias_variable([32])
    B2 = bias_variable([64])
    B3_FC1 = bias_variable([1024])
    B4_FC2 = bias_variable([image_labels])

    # CNN model
    X1 = tf.reshape(X, [-1, image_width, image_height, 1])  # shape=(?, 28, 28, 1)
    # Layer 1
    l1_conv = tf.nn.relu(conv2d(X1, W1) + B1)  # shape=(?, 28, 28, 32)
    l1_pool = max_pool_2x2(l1_conv)  # shape=(?, 14, 14, 32)
    l1_drop = tf.nn.dropout(l1_pool, drop_conv)
    # Layer 2
    l2_conv = tf.nn.relu(conv2d(l1_drop, W2) + B2)  # shape=(?, 14, 14, 64)
    l2_pool = max_pool_2x2(l2_conv)  # shape=(?, 7, 7, 64)
    l2_drop = tf.nn.dropout(l2_pool, drop_conv)
    # Layer 3 - FC1
    l3_flat = tf.reshape(l2_drop, [-1, W3_FC1.get_shape().as_list()[0]])  # shape=(?, 1024)
    l3_feed = tf.nn.relu(tf.matmul(l3_flat, W3_FC1) + B3_FC1)
    l3_drop = tf.nn.dropout(l3_feed, drop_hidden)
    # Layer 4 - FC2
    Y_pred = tf.nn.softmax(tf.matmul(l3_drop, W4_FC2) + B4_FC2)  # shape=(?, 10)
    predict = tf.argmax(Y_pred, 1)
    '''
    TensorFlow Session
    '''
    # start TensorFlow session
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    sess.run(init)
    saver.restore(sess, 'F:\PycharmProject\CNN_mnist_modify\model\my-model')
    # predict test set
    # using batches is more resource efficient
    predictvalue = np.zeros(test_images.shape[0])
    for i in range(0, test_images.shape[0]):
        imvalue = test_images[i]
        predictvalue[i] = predict.eval(feed_dict={X: [imvalue], drop_conv: DROPOUT_CONV, drop_hidden: DROPOUT_HIDDEN},
                                       session=sess)
    sess.close()
    return predictvalue


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
        count = 0
        for i in range(predint.shape[0]):
            if test_labels[i] == predint[i]:
                count += 1
        print("right %d" % (count))  # first value in list


if __name__ == "__main__":
    main(2)
