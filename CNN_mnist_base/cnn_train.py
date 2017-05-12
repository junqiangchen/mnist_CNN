"""A very simple CNN classifier.
Documentation at
http://niektemme.com/ @@to do

This script is based on the Tensoflow MNIST expert tutorial
See extensive documentation for the tutorial at
https://www.tensorflow.org/versions/master/tutorials/mnist/pros/index.html
"""

# import modules
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

# Parameters
LEARNING_RATE = 0.01
TRAINING_EPOCHS = 100
BATCH_SIZE = 200
DISPLAY_STEP = 10
DROPOUT_HIDDEN = 0.6
VALIDATION_SIZE = 2000  # Set to 0 to train on all available data

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


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


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


def batchnorm(Ylogits, n_out, is_test, convolutional=True):
    # adding the iteration prevents from averaging across non-existing iterations
    beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                       name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                        name='gamma', trainable=True)
    exp_moving_avg = tf.train.ExponentialMovingAverage(decay=0.5)
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


'''
Preprocessing for train dataset,data is csv type
'''
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

sess = tf.InteractiveSession()

# Create the model
x = tf.placeholder(tf.float32, [None, image_size])
y_ = tf.placeholder(tf.float32, [None, image_labels])
W = tf.Variable(tf.zeros([image_size, image_labels]))
b = tf.Variable(tf.zeros([image_labels]))
is_test = tf.placeholder(tf.bool)
keep_prob = tf.placeholder(tf.float32)

y = tf.nn.softmax(tf.matmul(x, W) + b)

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
conv1_bn = batchnorm(conv1, 32, is_test, True)
h_conv1 = tf.nn.relu(conv1_bn)
h_pool1 = max_pool_2x2(h_conv1)

conv2 = conv2d(h_pool1, W_conv2) + b_conv2
conv2_bn = batchnorm(conv2, 64, is_test, True)
h_conv2 = tf.nn.relu(conv2_bn)
h_pool2 = max_pool_2x2(h_conv2)

h_pool2_flat = tf.reshape(h_pool2, [-1, W_fc1.get_shape().as_list()[0]])
fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
fc1_bn = batchnorm(fc1, 1024, is_test, False)
h_fc1 = tf.nn.relu(fc1_bn)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Define loss and optimizer
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
"""
Train the model and save the model to disk as a my-model file
file is stored in the same directory as this python script is started

Based on the documentatoin at
https://www.tensorflow.org/versions/master/how_tos/variables/index.html
"""
epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]
saver = tf.train.Saver()
tf.summary.scalar("loss", cross_entropy)
tf.summary.scalar("accuracy", accuracy)
merged_summary_op = tf.summary.merge_all()
sess.run(tf.initialize_all_variables())
logs_path = "F:\PycharmProject\CNN_mnist_base\graph"
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
# with tf.Session() as sess:
# sess.run(init_op)
for i in range(TRAINING_EPOCHS):
    batch_x, batch_y = next_batch(BATCH_SIZE)
    _, summary = sess.run([train_step, merged_summary_op],
                          feed_dict={x: batch_x, y_: batch_y, is_test: False, keep_prob: DROPOUT_HIDDEN})
    summary_writer.add_summary(summary, i)
    if i % DISPLAY_STEP == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch_x, y_: batch_y, is_test: True, keep_prob: DROPOUT_HIDDEN})
        print("epochs %d, training accuracy %g" % (i, train_accuracy))
        validation_accuracy = accuracy.eval(feed_dict={x: validation_images[0:BATCH_SIZE],
                                                       y_: validation_labels[0:BATCH_SIZE], is_test: True,
                                                       keep_prob: DROPOUT_HIDDEN})
        print("epochs %d, validation accuracy %g" % (i, validation_accuracy))

im = Image.open('19.jpg')
im_array = np.array(im)
test_images = im_array.reshape((1, -1))
test_images = test_images.astype(np.float)
# convert from [0:255] => [0.0:1.0]
test_images = np.multiply(test_images, 1.0 / 255.0)
predict = tf.argmax(y_conv, 1)
predicted_lables = predict.eval(feed_dict={x: test_images, keep_prob: 1.0, is_test: False}, session=sess)
print(predicted_lables[0])
save_path = saver.save(sess, "F:\PycharmProject\CNN_mnist_base\model\my-model")
print("Model saved in file: ", save_path)
sess.close()
