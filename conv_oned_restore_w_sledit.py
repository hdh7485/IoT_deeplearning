import gzip
import os
import sys
import urllib.request, urllib.parse, urllib.error

import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy
import tensorflow as tf
import extract_server_sledit


import time
# data set parameter
RESF = 'result_person.txt'
GROUP = 'C1,C2,C3,C4,C5,C6,C7,C8,O1,O2,P1,P2,T1'
STATE = 'C1,C2,C3,C4,C5,C6,C7,C8,O1,O2,P1,P2,T1'
LABEL = '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14'

DOCTORS = ['00000000', '00000049', '00000063']
PATH = "/home/mskim/IoT/code/"

TRAIN_SIZE =        50000
TEST_SIZE =         1
VALIDATION_SIZE =   20000

BATCH_SIZE = 32
NUM_EPOCHS = 5

# model parameter
NUM_ROWS =  27
DATA_SIZE = 0

CONV_1_W = 1
CONV_1_H = 5
CONV_1_PADDING = 'VALID'

CONV_2_W = 1
CONV_2_H = 5

CONV_3_W = 1
CONV_3_H = 5

CONV_1_D =  32
CONV_2_D =  64
CONV_3_D =  128
FULL_D =    1152


POOL_1 = 2
POOL_2 = 2
POOL_3 = 2

POOL_1_k = 3
POOL_2_k = 3
POOL_3_k = 3

# user inputs
SEED = 66478  # Set to None for random seed.
GROUPS = []
STATES = []
NUM_CHANNELS = 27
NUM_LABELS = 0

tf.app.flags.DEFINE_string('resf', RESF, "Set the result file name")
tf.app.flags.DEFINE_boolean("self_test", False, "True if running a self test.")
tf.app.flags.DEFINE_string('train_dir', '/home/mskim/new_test_helped/train_dir',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('groups', GROUP, "Set the groups")
tf.app.flags.DEFINE_string('states', STATE, "Set the states")
tf.app.flags.DEFINE_string('labels', LABEL, "Set the labels")

tf.app.flags.DEFINE_integer('train_size',   TRAIN_SIZE, "Set")
tf.app.flags.DEFINE_integer('test_size',    TEST_SIZE, "Set")
tf.app.flags.DEFINE_integer('val_size',     VALIDATION_SIZE, "Set")

tf.app.flags.DEFINE_integer('batch_size', BATCH_SIZE, "Set")
tf.app.flags.DEFINE_integer('num_epochs', NUM_EPOCHS, "Set")

tf.app.flags.DEFINE_integer('num_rows', NUM_ROWS, "Set")
tf.app.flags.DEFINE_integer('data_size',DATA_SIZE, "Set")

tf.app.flags.DEFINE_integer('c1w', CONV_1_W, "Set")
tf.app.flags.DEFINE_integer('c1h', CONV_1_H, "Set")

tf.app.flags.DEFINE_integer('c2w', CONV_2_W, "Set")
tf.app.flags.DEFINE_integer('c2h', CONV_2_H, "Set")

tf.app.flags.DEFINE_integer('c1d', CONV_1_D, "Set")
tf.app.flags.DEFINE_integer('c2d', CONV_2_D, "Set")
tf.app.flags.DEFINE_integer('fd', FULL_D, "Set")

tf.app.flags.DEFINE_integer('p1', POOL_1, "Set")
tf.app.flags.DEFINE_integer('p2', POOL_2, "Set")

FLAGS = tf.app.flags.FLAGS

res_file_name =     FLAGS.resf
# data set parameter
TRAIN_SIZE =        FLAGS.train_size
TEST_SIZE =         FLAGS.test_size
VALIDATION_SIZE =   FLAGS.val_size

BATCH_SIZE = FLAGS.batch_size
NUM_EPOCHS = FLAGS.num_epochs

# model parameter
NUM_ROWS =  FLAGS.num_rows
DATA_SIZE = FLAGS.data_size

CONV_1_W = FLAGS.c1w
CONV_1_H = FLAGS.c1h

CONV_2_W = FLAGS.c2w
CONV_2_H = FLAGS.c2h

CONV_1_D =  FLAGS.c1d
CONV_2_D =  FLAGS.c2d
FULL_D =    FLAGS.fd

POOL_1 = FLAGS.p1
POOL_2 = FLAGS.p2


def fake_data(num_images):
    """Generate a fake dataset that matches the dimensions of MNIST."""
    NUM_LABELS = 8
    data = numpy.ndarray(
        shape=(num_images, NUM_ROWS, DATA_SIZE, NUM_CHANNELS),
        dtype=numpy.float32)
    labels = numpy.zeros(shape=(num_images, NUM_LABELS), dtype=numpy.float32)
    for image in range(num_images):
        label = image % 2
        data[image, :, :, 0] = label - 0.5
        labels[image, label] = 1.0

    return data, labels


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
        100.0 *
        numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) /
        predictions.shape[0])

def correct_detection(predictions, labels):
    signal = numpy.ones(labels.shape[0])
    normal = numpy.zeros(labels.shape[0])
    return (
        100.0 *
        numpy.sum(numpy.logical_and((numpy.argmax(labels, 1) == signal), numpy.argmax(predictions, 1) == signal)) /
        numpy.sum(numpy.argmax(labels, 1) == signal))

def false_alarm(predictions, labels):
    signal = numpy.ones(labels.shape[0])
    normal = numpy.zeros(labels.shape[0])

    return (
        100.0 *
        numpy.sum(numpy.logical_and((numpy.argmax(labels, 1) == normal), numpy.argmax(predictions, 1) == signal)) /
        numpy.sum( numpy.argmax(labels, 1) == normal) )

def xavier_init(n_inputs, n_outputs, uniform=False):

    if uniform:
        # 6 was used in the paper.
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        # 3 gives us approximately the same limits as above since this repicks
        # values greater than 2 standard deviations from the mean.
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

def xavier_init2(n_inputs, n_outputs2,n_outputs3, uniform=False):

    if uniform:
        # 6 was used in the paper.
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs2*n_outputs3))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        # 3 gives us approximately the same limits as above since this repicks
        # values greater than 2 standard deviations from the mean.
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs2*n_outputs3))
        return tf.truncated_normal_initializer(stddev=stddev)

def main(argv=None):  # pylint: disable=unused-argument
    if FLAGS.self_test:
        print('Running self-test.')
        train_data, train_labels = fake_data(256)
        validation_data, validation_labels = fake_data(16)
        test_data, test_labels = fake_data(256)
        num_epochs = 1
    else:
        # Extract it into numpy arrays.
        STATES = FLAGS.states.split(',')
        LABELS = FLAGS.labels.split(',')

        NUM_LABELS = len(set(LABELS))


        # test_data_list = []
        # test_data = extract_server.extract_data_oned(numRows=NUM_ROWS, numData=TEST_SIZE, states=STATES, labels=LABELS, mode='test',DATA_SIZE = DATA_SIZE,NUM_CHANNELS=NUM_CHANNELS, ONED=True)
        # test_data = test_data[:, :, 0, :]
        # # print "test_data", numpy.shape(test_data)
        # test_data_list.append(test_data)



    conv1_weights = tf.get_variable("C1", shape=[ CONV_1_H, NUM_CHANNELS, CONV_1_D], initializer=xavier_init2(CONV_1_H,  NUM_CHANNELS, CONV_1_D))
    conv1_biases = tf.Variable(tf.zeros([CONV_1_D]))

    conv2_weights = tf.get_variable("C2", shape=[ CONV_2_H, CONV_1_D, CONV_2_D],
                                    initializer=xavier_init2(CONV_2_H,  CONV_1_D, CONV_2_D))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[CONV_2_D]))

    conv3_weights = tf.get_variable("C3", shape=[CONV_3_H, CONV_2_D, CONV_3_D],
                                    initializer=xavier_init2(CONV_3_H,  CONV_2_D, CONV_3_D))
    conv3_biases = tf.Variable(tf.constant(0.1, shape=[CONV_3_D]))

    if CONV_1_PADDING == 'VALID':
        # WH = (NUM_ROWS - CONV_1_W + 2) / (POOL_1 * POOL_2 ) * (DATA_SIZE - CONV_1_H + 1) / (1 * 1) * CONV_2_D ################################################################
        WH=128*3
    else:
        WH = (NUM_ROWS) / (POOL_1 * POOL_2) * (DATA_SIZE) / (1 * 1) * CONV_2_D

    fc1_weights = tf.get_variable("W1", shape=[WH, FULL_D], initializer=xavier_init(WH, FULL_D))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[FULL_D]))
    fc2_weights = tf.get_variable("W2", shape=[FULL_D, NUM_LABELS], initializer=xavier_init(FULL_D, NUM_LABELS))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))


    def model(data, train):
        # print 'data ', numpy.shape(data)
        conv = tf.nn.conv1d(data, conv1_weights, stride=1,
                            padding="SAME")
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))

        if True:
            pool = tf.layers.max_pooling1d(relu, POOL_1, strides=POOL_1, padding='VALID')
        else:
            pool = relu

        conv = tf.nn.conv1d(pool, conv2_weights, stride=1, padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))

        if True:
            pool = tf.layers.max_pooling1d(relu, POOL_2, strides=POOL_2, padding='VALID')
        else:
            pool = relu

        conv = tf.nn.conv1d(pool, conv3_weights, stride=1, padding="SAME")
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv3_biases))

        if True:
            pool = tf.layers.max_pooling1d(relu, POOL_3, strides=POOL_3, padding='VALID')
        else:
            pool = relu

        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(
            pool,
            [pool_shape[0], pool_shape[1] * pool_shape[2]])

        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        return tf.matmul(hidden, fc2_weights) + fc2_biases

    saver = tf.train.Saver(tf.all_variables())
    with tf.Session() as s:
        # Run all the initializers to prepare the trainable parameters.
        tf.initialize_all_variables().run()
        saver.restore(s, PATH + 'w_0322.ckpt')
        while True:
            for did in DOCTORS:
                test_data_list = []
                test_data = extract_server_sledit.extract_data_oned(did, numRows=NUM_ROWS, numData=TEST_SIZE,  states=STATES, labels=LABELS, mode='test', DATA_SIZE=DATA_SIZE, NUM_CHANNELS=NUM_CHANNELS,ONED=True)
                test_data = test_data[:, :, 0, :]
                # print "test_data", numpy.shape(test_data)
                test_data_list.append(test_data)
                test_data_node = tf.constant(test_data_list[0])
                test_prediction = tf.nn.softmax(model(test_data_node, False))
                p_num = numpy.argmax(test_prediction.eval(),1)
                e=STATES[p_num[0]]
                # print e
                extract_server_sledit.save_to_server(did, e)
                # print did
if __name__ == '__main__':
    tf.app.run()
