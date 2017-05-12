import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
# download and install the MNIST data automatically
from tensorflow.examples.tutorials.mnist import input_data
# slim = tf.contrib.slim
import tensorflow.contrib.slim as slim

mnist = input_data.read_data_sets("data/", one_hot=True)


def add_batch_norm_layer(l, x, phase_train, trainable=True, scope='BN'):
    bn_layer = batch_norm_layer(x, phase_train, scope_bn=scope + l, trainable=trainable)
    return bn_layer


def batch_norm_layer(x, phase_train, scope_bn, trainable=True):
    bn_train = batch_norm(x, decay=0.999, center=True, scale=True,
                          updates_collections=None,
                          is_training=True,
                          reuse=None,  # is this right?
                          trainable=trainable,
                          scope=scope_bn)
    bn_inference = batch_norm(x, decay=0.999, center=True, scale=True,
                              updates_collections=None,
                              is_training=False,
                              reuse=True,  # is this right?
                              trainable=trainable,
                              scope=scope_bn)
    z = tf.cond(phase_train, lambda: bn_train, lambda: bn_inference)
    return z


def get_NN_layer(l, x, dims, init, trainable_bn, phase_train=None, scope="NNLayer"):
    init_W, init_b = init
    dim_input, dim_out = dims
    with tf.name_scope(scope + l):
        W = tf.get_variable(name='W' + l, dtype=tf.float32, initializer=init_W, regularizer=None, trainable=True,
                            shape=[dim_input, dim_out])
        b = tf.get_variable(name='b' + l, dtype=tf.float32, initializer=init_b, regularizer=None, trainable=True)
        with tf.name_scope('Z' + l):
            z = tf.matmul(x, W) + b
            if phase_train is not None:
                z = add_batch_norm_layer(l, z, phase_train, trainable_bn)
        with tf.name_scope('A' + l):
            a = tf.nn.relu(z)  # (M x D1) = (M x D) * (D x D1)
            # a = tf.sigmoid(z)
    return a


def softmax_layer(l, x, dims, init):
    init_W, init_b = init
    dim_input, dim_out = dims
    with tf.name_scope('Z' + l):
        W = tf.get_variable(name='W' + l, dtype=tf.float32, initializer=init_W, regularizer=None, trainable=True,
                            shape=[dim_input, dim_out])
        b = tf.get_variable(name='b' + l, dtype=tf.float32, initializer=init_b, regularizer=None, trainable=True)
        z = tf.matmul(x, W) + b
    with tf.name_scope('y'):
        y = tf.nn.softmax(tf.matmul(x, W) + b)
    return y


###
###

def build_NN_two_hidden_layers(x, phase_train, trainable_bn):
    ## first layer
    [D_in, D_out] = [784, 50]
    init_W, init_b = tf.contrib.layers.xavier_initializer(dtype=tf.float32), tf.constant(0.1, shape=[D_out])
    A1 = get_NN_layer(l='1', x=x, dims=[D_in, D_out], init=(init_W, init_b), trainable_bn=trainable_bn,
                      phase_train=phase_train, scope="NNLayer")
    ## second layer
    [D_in, D_out] = [50, 49]
    init_W, init_b = tf.contrib.layers.xavier_initializer(dtype=tf.float32), tf.constant(0.1, shape=[D_out])
    A2 = get_NN_layer(l='2', x=A1, dims=[D_in, D_out], init=(init_W, init_b), trainable_bn=trainable_bn,
                      phase_train=phase_train, scope="NNLayer")
    ## final layer
    [D_in, D_out] = [49, 10]
    init_W, init_b = tf.contrib.layers.xavier_initializer(dtype=tf.float32), tf.constant(0.1, shape=[D_out])
    y = softmax_layer(l='3', x=A2, dims=[D_in, D_out], init=(init_W, init_b))
    return y


##

def get_batch_feed(M, phase_train, x, y_):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    feed_dict = {x: batch_xs, y_: batch_ys, phase_train: True} if (phase_train is not None) else {x: batch_xs,
                                                                                                  y_: batch_ys}
    return feed_dict


def get_feed_for_learning(phase_train, X_train, X_cv, X_test, Y_train, Y_cv, Y_test, x, y_):
    '''
        Returns the feed_dict for sess.run when BN is present or not.
    '''
    if phase_train is not None:  # DO BN
        feed_dict_train, feed_dict_cv, feed_dict_test = {x: X_train, y_: Y_train, phase_train: False}, {x: X_cv,
                                                                                                        y_: Y_cv,
                                                                                                        phase_train: False}, {
                                                            x: X_test, y_: Y_test, phase_train: False}
    else:  # Don't do BN
        feed_dict_train, feed_dict_cv, feed_dict_test = {x: X_train, y_: Y_train}, {x: X_cv, y_: Y_cv}, {x: X_test,
                                                                                                         y_: Y_test}
    return feed_dict_train, feed_dict_cv, feed_dict_test


def get_MNIST_data_sets():
    X_train, Y_train = mnist.train.images, mnist.train.labels
    X_cv, Y_cv = mnist.validation.images, mnist.validation.labels
    X_test, Y_test = mnist.test.images, mnist.test.labels
    return X_train, X_cv, X_test, Y_train, Y_cv, Y_test


def build_NN_two_hidden_layers_sguada(x, is_training):
    batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
    with slim.arg_scope([slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.constant_initializer(0.1),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        net = slim.fully_connected(x, 50, scope='A1')
        net = slim.fully_connected(net, 49, scope='A2')
        y = slim.fully_connected(net, 10, activation_fn=tf.nn.softmax, normalizer_fn=None, scope='A3')
    return y


##

def main():
    ##BN ON or OFF
    bn = True
    trainable_bn = True
    phase_train = tf.placeholder(tf.bool, name='phase_train') if bn else  None
    ##
    x = tf.placeholder(tf.float32, [None, 784])
    y = build_NN_two_hidden_layers(x, phase_train, trainable_bn)
    # is_training = True
    # y = build_NN_two_hidden_layers_sguada(x, is_training=is_training)
    ### training
    # new placeholder to input the correct answers
    y_ = tf.placeholder(tf.float32, [None, 10])
    # loss function
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    # single training step opt.
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    # init op
    init = tf.initialize_all_variables()
    # launch model in a session
    sess = tf.Session()
    # run opt to initialize
    sess.run(init)

    X_train, X_cv, X_test, Y_train, Y_cv, Y_test = get_MNIST_data_sets()
    feed_dict_train, feed_dict_cv, feed_dict_test = get_feed_for_learning(phase_train, X_train, X_cv, X_test, Y_train,
                                                                          Y_cv, Y_test, x, y_)

    # we'll run the training step 1000 times
    for i in range(1000):
        # batch_xs, batch_ys = mnist.train.next_batch(100)
        M = 100
        batch_feed = get_batch_feed(M, phase_train, x, y_)
        sess.run(train_step, feed_dict=batch_feed)

    # list of booleans indicating correct predictions
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run(accuracy, feed_dict=feed_dict_test))


if __name__ == '__main__':
    main()
