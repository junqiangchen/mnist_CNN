"""Predict a handwritten integer (MNIST expert).

Script requires
1) saved model (model2.ckpt file) in the same location as the script is run from.
(requried a model created in the MNIST expert tutorial)
2) one argument (png file location of a handwritten integer)

Documentation at:
http://niektemme.com/ @@to do
"""

# import modules
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

image_size = 784
image_width = image_height = 28
image_labels = 10


# Weight initialization (Xavier's init)
def weight_xavier_init(n_inputs, n_outputs, uniform=True):
    if uniform:
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def batchnorm(Ylogits, offset, is_test, convolutional=True):
    # adding the iteration prevents from averaging across non-existing iterations
    exp_moving_avg = tf.train.ExponentialMovingAverage(decay=0.999)
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def predictint(test_images):
    """
    This function returns the predicted integer.
    The imput is the pixel values from the imageprepare() function.
    """
    # Define the model (same as when creating the model file)
    x = tf.placeholder(tf.float32, [None, image_size])
    W = tf.Variable(tf.zeros([image_size, image_labels]))
    b = tf.Variable(tf.zeros([image_labels]))
    is_test = tf.placeholder(tf.bool)
    # Model Parameters
    W_conv1 = tf.get_variable("W_conv1", shape=[5, 5, 1, 32], initializer=weight_xavier_init(5 * 5 * 1, 32))
    W_conv2 = tf.get_variable("W_conv2", shape=[5, 5, 32, 64], initializer=weight_xavier_init(5 * 5 * 32, 64))
    W_fc1 = tf.get_variable("W_fc1", shape=[64 * 7 * 7, 1024], initializer=weight_xavier_init(64 * 7 * 7, 1024))
    W_fc2 = tf.get_variable("W_fc2", shape=[1024, image_labels], initializer=weight_xavier_init(1024, image_labels))

    b_conv1 = bias_variable([32])
    b_conv2 = bias_variable([64])
    b_fc1 = bias_variable([1024])
    b_fc2 = bias_variable([image_labels])

    x_image = tf.reshape(x, [-1, image_width, image_height, 1])
    conv1 = conv2d(x_image, W_conv1) + b_conv1
    conv1_bn = batchnorm(conv1, b_conv1, is_test, True)
    h_conv1 = tf.nn.relu(conv1_bn)
    h_pool1 = max_pool_2x2(h_conv1)

    conv2 = conv2d(h_pool1, W_conv2) + b_conv2
    conv2_bn = batchnorm(conv2, b_conv2, is_test, True)
    h_conv2 = tf.nn.relu(conv2_bn)
    h_pool2 = max_pool_2x2(h_conv2)

    h_pool2_flat = tf.reshape(h_pool2, [-1, W_fc1.get_shape().as_list()[0]])
    fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    fc1_bn = batchnorm(fc1, b_fc1, is_test, False)
    h_fc1 = tf.nn.relu(fc1_bn)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    init_op = tf.initialize_all_variables()
    saver = tf.train.Saver()

    """
    Load the my-model file
    file is stored in the same directory as this python script is started
    Use the model to predict the integer. Integer is returend as list.

    Based on the documentatoin at
    https://www.tensorflow.org/versions/master/how_tos/variables/index.html
    """
    predicted_lables = np.zeros(test_images.shape[0])
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, "F:\PycharmProject\CNN_mnist_base\model\my-model")
        # print ("Model restored.")
        predict = tf.argmax(y_conv, 1)
        for i in range(0, test_images.shape[0]):
            imagein = test_images[i]
            predicted_lables[i] = predict.eval(feed_dict={x: [imagein], keep_prob: 1.0, is_test: False}, session=sess)
        sess.close()
        return predicted_lables


def imagecsvprepare(argv):
    test_imagesdata = pd.read_csv(argv)
    test_images = test_imagesdata.iloc[:, 1:].values
    test_images = test_images.astype(np.float)
    # convert from [0:255] => [0.0:1.0]
    test_images = np.multiply(test_images, 1.0 / 255.0)
    print('test_images({0[0]},{0[1]})'.format(test_images.shape))
    return test_images


def imageprepare(argv):
    im = Image.open(argv)
    im_array = np.array(im)
    test_images = im_array.reshape((1, -1))
    test_images = test_images.astype(np.float)
    # convert from [0:255] => [0.0:1.0]
    test_images = np.multiply(test_images, 1.0 / 255.0)
    print('test_images({0[0]},{0[1]})'.format(test_images.shape))
    return test_images


def main():
    """
    Main function.
    """
    imvalue = imageprepare('19.jpg')
    predint = predictint(imvalue)
    for i in range(predint.shape[0]):
        print(predint[i])  # first value in list


if __name__ == "__main__":
    main()
