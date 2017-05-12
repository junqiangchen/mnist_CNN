'''
2-D Convolutional Neural Networks using TensorFlow library for jieheganjun detection.
Author: junqiangChen

Train instances: 50000 number images with vector format (1 number = 1 x 1024)
Test instances: 10000 number images with vector format  (1 number = 1 x 1024)
'''

import numpy as np
import pandas as pd
import tensorflow as tf

# Parameters
LEARNING_RATE = 0.0001
TRAINING_EPOCHS = 10000
BATCH_SIZE = 200
DISPLAY_STEP = 10
DROPOUT_CONV = 0.8
DROPOUT_HIDDEN = 0.7
VALIDATION_SIZE = 2000  # Set to 0 to train on all available data

image_size = 1024
image_width = image_height = 32
image_labels = 2


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


# 2D convolution
def conv2d(X, W):
    return tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')


# Max Pooling
def max_pool_2x2(X):
    return tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


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


def training(cost, learning_rate):
    optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op


def evaluation(Y_gt, Y_pred):
    correct_predict = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y_gt, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict, 'float'))
    return accuracy


def mlogloss(predicted, actual):
    '''

    :param predicted: predicted probability
    :param actual: actual value,label
    :return:Logistic Regression value
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
    ans = -1 * mysum / (nrow + 1)
    return ans


'''
Preprocessing for dataset
'''
# Read data set (Train data from CSV file)
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

'''
Create model with 2D CNN
'''
# Create Input and Output
X = tf.placeholder('float', shape=[None, image_size])  # mnist data image of shape 32*32=1024
Y_gt = tf.placeholder('float', shape=[None, image_labels])  # 0-1 => 2 classes
lr = tf.placeholder('float')
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

# Cost function and training 
cost = -tf.reduce_sum(Y_gt * tf.log(Y_pred))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
correct_predict = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y_gt, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, 'float'))
predict = tf.argmax(Y_pred, 1)

'''
TensorFlow Session
'''
# start TensorFlow session
init = tf.initialize_all_variables()
saver = tf.train.Saver(tf.all_variables())
tf.summary.scalar("loss", cost)
tf.summary.scalar("accuracy", accuracy)
for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)
merged_summary_op = tf.summary.merge_all()
sess = tf.InteractiveSession()
logs_path = "F:\PycharmProject\CNN_jieheganjun2017.3.8_NoBN\graph"
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
sess.run(init)

# visualisation variables
train_accuracies = []
validation_accuracies = []

DISPLAY_STEP = 1
epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]
for i in range(TRAINING_EPOCHS):
    # get new batch
    batch_xs, batch_ys = next_batch(BATCH_SIZE)

    # check progress on every 1st,2nd,...,10th,20th,...,100th... step
    if i % DISPLAY_STEP == 0 or (i + 1) == TRAINING_EPOCHS:
        train_accuracy = accuracy.eval(feed_dict={X: batch_xs[BATCH_SIZE // 10:],
                                                  Y_gt: batch_ys[BATCH_SIZE // 10:],
                                                  lr: LEARNING_RATE,
                                                  drop_conv: DROPOUT_CONV,
                                                  drop_hidden: DROPOUT_HIDDEN})
        if VALIDATION_SIZE:
            validation_accuracy = accuracy.eval(
                feed_dict={X: batch_xs[0:BATCH_SIZE // 10], Y_gt: batch_ys[0:BATCH_SIZE // 10], lr: LEARNING_RATE,
                           drop_conv: DROPOUT_CONV, drop_hidden: DROPOUT_HIDDEN})
            print('epochs %d training_accuracy / validation_accuracy => %.2f / %.2f ' % (
                i, train_accuracy, validation_accuracy))

            validation_accuracies.append(validation_accuracy)

        else:
            print('epochs %d batch_training_accuracy => %.4f' % (i, train_accuracy))
            train_accuracies.append(train_accuracy)  # increase DISPLAY_STEP
        if i % (DISPLAY_STEP * 10) == 0 and i:
            DISPLAY_STEP *= 10
            # train on batch
    _, summary = sess.run([train_op, merged_summary_op],
                          feed_dict={X: batch_xs, Y_gt: batch_ys, lr: LEARNING_RATE,
                                     drop_conv: DROPOUT_CONV,
                                     drop_hidden: DROPOUT_HIDDEN})
    summary_writer.add_summary(summary, i)

# check final accuracy on validation set
if VALIDATION_SIZE:
    validation_accuracy = accuracy.eval(feed_dict={X: validation_images,
                                                   Y_gt: validation_labels,
                                                   lr: LEARNING_RATE,
                                                   drop_conv: DROPOUT_CONV, drop_hidden: DROPOUT_HIDDEN})
    pred = Y_pred.eval(feed_dict={X: validation_images,
                                  Y_gt: validation_labels,
                                  lr: LEARNING_RATE,
                                  drop_conv: DROPOUT_CONV, drop_hidden: DROPOUT_HIDDEN})
print('all_validation_accuracy => %.4f' % validation_accuracy)
print('all_validation_log_loss+>%.4f' % mlogloss(pred, validation_labels))

save_path = saver.save(sess, 'F:\PycharmProject\CNN_jieheganjun2017.3.8_NoBN\model\mymodel')
print("Model saved in file:", save_path)
sess.close()
