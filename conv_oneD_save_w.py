"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.8%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to exectute a short self-test.
"""

# edited by SJLIM
# edited by SJLIM
import gzip
import os
import sys
import urllib.request, urllib.parse, urllib.error

import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy
import tensorflow as tf
import extractData


import time

# saver = tf.train.Saver()
# data set parameter
RESF = 'result_person_IMU.txt'
# GROUP = 'C1,C2,C3,C4,C5,C6,C7,C8,O1,O2,P1,P2,T1'
# STATE = 'C1,C2,C3,C4,C5,C6,C7,C8,O1,O2,P1,P2,T1'
GROUP = 'C1,C2,C3,C4,C5,C6,C7,C8,O1,O2,P1,P2,T1'
STATE = 'C1,C2,C3,C4,C5,C6,C7,C8,O1,O2,P1,P2,T1,mC1,mC2,mC3,mC4,mC5,mC6,mC7,mC8,mO1,mO2,mP1,mP2,mT1'
LABEL = '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25'
DATES = '20170331,20170403,20170404,20170405,20170406,20170407,20170410,20170411,20170412,20170413,20170414,20170417,20170419,20170420,20170421,20170424,20170425,20170426,20170427,20170428'
MOVE = 'C1_C2,C1_C3,C1_C4,C1_C5,C1_C6,C1_C7,C1_C8,C1_O1,C1_O2,C1_P1,C1_P2,C1_T1,C1_Z1,C1_A1,C1_A2,' \
       'C2_C1,C2_C3,C2_C4,C2_C5,C2_C6,C2_C7,C2_C8,C2_O1,C2_O2,C2_P1,C2_P2,C2_T1,C2_Z1,C2_A1,C2_A2,' \
       'C3_C1,C3_C2,C3_C4,C3_C5,C3_C6,C3_C7,C3_C8,C3_O1,C3_O2,C3_P1,C3_P2,C3_T1,C3_Z1,C3_A1,C3_A2,' \
       'C4_C1,C4_C2,C4_C3,C4_C5,C4_C6,C4_C7,C4_C8,C4_O1,C4_O2,C4_P1,C4_P2,C4_T1,C4_Z1,C4_A1,C4_A2, ' \
       'C5_C1,C5_C2,C5_C3,C5_C4,C5_C6,C5_C7,C5_C8,C5_O1,C5_O2,C5_P1,C5_P2,C5_T1,C5_Z1,C5_A1,C5_A2, ' \
       'C6_C1,C6_C2,C6_C3,C6_C4,C6_C5,C6_C7,C6_C8,C6_O1,C6_O2,C6_P1,C6_P2,C6_T1,C6_Z1,C6_A1,C6_A2, ' \
       'C7_C1,C7_C2,C7_C3,C7_C4,C7_C5,C7_C6,C7_C8,C7_O1,C7_O2,C7_P1,C7_P2,C7_T1,C7_Z1,C7_A1,C7_A2,' \
       'C8_C1,C8_C2,C8_C3,C8_C4,C8_C5,C8_C6,C8_C7,C8_O1,C8_O2,C8_P1,C8_P2,C8_T1,C8_Z1,C8_A1,C8_A2,' \
       'O1_C1,O1_C2,O1_C3,O1_C4,O1_C5,O1_C6,O1_C7,O1_C8,O1_O2,O1_P1,O1_P2,O1_T1,O1_Z1,O1_A1,O1_A2,' \
       'O2_C1,O2_C2,O2_C3,O2_C4,O2_C5,O2_C6,O2_C7,O2_C8,O2_O1,O2_P1,O2_P2,O2_T1,O2_Z1,O2_A1,O2_A2,' \
       'P1_C1,P1_C2,P1_C3,P1_C4,P1_C5,P1_C6,P1_C7,P1_C8,P1_O1,P1_O2,P1_P2,P1_T1,P1_Z1,P1_A1,P1_A2,' \
       'P2_C1,P2_C2,P2_C3,P2_C4,P2_C5,P2_C6,P2_C7,P2_C8,P2_O1,P2_O2,P2_P1,P2_T1,P2_Z1,P2_A1,P2_A2,' \
       'T1_C1,T1_C2,T1_C3,T1_C4,T1_C5,T1_C6,T1_C7,T1_C8,T1_O1,T1_O2,T1_P1,T1_P2,T1_Z1,T1_A1,T1_A2,' \
       'Z1_C1,Z1_C2,Z1_C3,Z1_C4,Z1_C5,Z1_C6,Z1_C7,Z1_C8,Z1_O1,Z1_O2,Z1_P1,Z1_P2,Z1_T1,Z1_A1,Z1_A2,' \
       'A1_C1,A1_C2,A1_C3,A1_C4,A1_C5,A1_C6,A1_C7,A1_C8,A1_O1,A1_O2,A1_P1,A1_P2,A1_T1,A1_Z1,A1_A2,' \
       'A2_C1,A2_C2,A2_C3,A2_C4,A2_C5,A2_C6,A2_C7,A2_C8,A2_O1,A2_O2,A2_P1,A2_P2,A2_T1,A2_Z1,A2_A1'
Doctor = '00000004,00000049'
# LABEL = '0,1'
TRAIN_SIZE =        50000
TEST_SIZE =         20000
VALIDATION_SIZE =   20000

BATCH_SIZE = 32
NUM_EPOCHS = 5

# d_rate = 0.20

# model parameter
NUM_ROWS =  27
DATA_SIZE = 0

CONV_1_W = 1
CONV_1_H = 5
CONV_1_PADDING = 'VALID'

CONV_2_W = 1
CONV_2_H = 3

CONV_3_W = 1
CONV_3_H = 2

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
NUM_CHANNELS = 47
NUM_LABELS = 0

tf.app.flags.DEFINE_string('resf', RESF, "Set the result file name")
tf.app.flags.DEFINE_boolean("self_test", False, "True if running a self test.")
tf.app.flags.DEFINE_string('train_dir', '/home/mskim/train_dir',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('dates', DATES, "Set the dates")
tf.app.flags.DEFINE_string('moves', MOVE, "Set the moves")
tf.app.flags.DEFINE_string('doctor', Doctor, "Set the doctor")
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
    print('predictions')
    print(numpy.argmax(predictions,1))
    print('labels')
    print(numpy.argmax(labels,1))
    return 100.0 - (
        100.0 *
        numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) /
        predictions.shape[0])

def correct_detection(predictions, labels):
    signal = numpy.ones(labels.shape[0])
    normal = numpy.zeros(labels.shape[0])

    # return (
    #     100.0 *
    #     numpy.sum(numpy.logical_and((numpy.argmax(labels, 1) == signal), numpy.argmax(predictions, 1) == signal)) /
    #     numpy.sum(numpy.argmax(labels, 1) == signal))
    return (
        100.0 *
        numpy.sum(numpy.logical_and((numpy.argmax(labels, 1) == signal), numpy.argmax(predictions, 1) == signal)) /
        numpy.sum(numpy.argmax(labels, 1) == signal))


def error_draw_matrix_rate(predictions, labels, matrix):
    """Return the error rate based on dense predictions and 1-hot labels."""
    print('predictions')
    print(numpy.shape(predictions))
    print('labels')
    print(numpy.shape(labels))
    p=numpy.argmax(predictions, 1)
    l=numpy.argmax(labels, 1)
    i=len(predictions)-1#########################################################################################################################################################

    while (True):
        matrix [p[i]-1][l[i]-1]=matrix[p[i]-1][l[i]-1]+1
        i = i - 1
        if i < 0:
            break

    rate = 100.0 - (
                100.0 *
                numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) /
                predictions.shape[0])

    return rate, matrix#########################################################################################################################################################



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
        DATES = FLAGS.dates.split(',')
        DOCTORS = FLAGS.doctors.split(',')
        MOVES = FLAGS.moves.split(',')
        LABELS = FLAGS.labels.split(',')
        print(STATES)
        print(MOVES)
        print(DATES)
        print(LABELS)
        NUM_LABELS = len(set(LABELS))
        print(NUM_LABELS)

        train_data, train_labels = extractData.extract_data_oned(numRows = NUM_ROWS, numData = TRAIN_SIZE, moves=MOVES, doctors=DOCTORS, dates=DATES,states = STATES, labels = LABELS, mode = 'train', DATA_SIZE = DATA_SIZE,NUM_CHANNELS=NUM_CHANNELS,ONED=True)
        train_data = train_data[:,:,0,:]
        print("train_data", numpy.shape(train_data))
        #print " change train_data", numpy.shape(train_data[:,:,0,:])
        # print 'train_data[0]'
        # print train_data[0]
        # print 'train_data[1]'
        # print train_data[1]
        test_data_list = []
        test_labels_list = []



        test_data, test_labels = extractData.extract_data_oned(numRows=NUM_ROWS, numData=TEST_SIZE, states=STATES, moves=MOVES, doctors=DOCTORS, dates=DATES,labels=LABELS, mode='test',DATA_SIZE = DATA_SIZE,NUM_CHANNELS=NUM_CHANNELS, ONED=True)
        test_data = test_data[:, :, 0, :]
        print("test_data", numpy.shape(test_data))
        test_data_list.append(test_data)
        test_labels_list.append(test_labels)



        validation_data, validation_labels = extractData.extract_data_oned(numRows = NUM_ROWS, numData = VALIDATION_SIZE, states = STATES, moves=MOVES, doctors=DOCTORS, dates=DATES,labels = LABELS, mode = 'validate',DATA_SIZE = DATA_SIZE,NUM_CHANNELS=NUM_CHANNELS, ONED=True)
        validation_data = validation_data[:, :, 0, :]
        print("validation_data", numpy.shape(validation_data))
        # print 'test_data1'
        # print test_data
        #
        # print 'test_labels1'
        # print test_labels

        #train_data, train_labels = extractData.extract_data(numRows = NUM_ROWS, numData = 10000, mode = 'train')
        #test_data, test_labels = extractData.extract_data(numRows = NUM_ROWS, numData = 2000, mode = 'test')
        #validation_data, validation_labels = extractData.extract_data(numRows = NUM_ROWS, numData = 2000, mode = 'validate')
        # Generate a validation set.
        num_epochs = NUM_EPOCHS


    if gfile.Exists(FLAGS.train_dir):
        gfile.DeleteRecursively(FLAGS.train_dir)
    gfile.MakeDirs(FLAGS.train_dir)

    train_size = train_labels.shape[0]

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    # train_data_node = tf.placeholder(
    #     tf.float32,
    #     shape=(BATCH_SIZE, NUM_ROWS, DATA_SIZE, NUM_CHANNELS))
    train_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, NUM_ROWS,  NUM_CHANNELS))
    print("train_data_node", train_data_node)
    print("train_data_node", train_data_node[0])
    train_labels_node = tf.placeholder(tf.float32,
                                       shape=(BATCH_SIZE, NUM_LABELS))

    # For the validation and test data, we'll just hold the entire dataset in
    # one constant node.
    print("validation_data", numpy.shape(validation_data))
    validation_data_node = tf.constant(validation_data)
    print("validation_data_node", validation_data_node)
    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when when we call:
    # {tf.initialize_all_variables().run()}
    # conv1_weights = tf.Variable(
    #     tf.truncated_normal([CONV_1_W, CONV_1_H, NUM_CHANNELS, CONV_1_D], # 5x5 filter, depth 32.
    #                         stddev=0.1,
    #                         seed=SEED))
    conv1_weights = tf.get_variable("C1", shape=[ CONV_1_H, NUM_CHANNELS, CONV_1_D], initializer=xavier_init2(CONV_1_H,  NUM_CHANNELS, CONV_1_D))
    conv1_biases = tf.Variable(tf.zeros([CONV_1_D]))


    # conv2_weights = tf.Variable(
    #     tf.truncated_normal([CONV_2_W, CONV_2_H, CONV_1_D, CONV_2_D],
    #                         stddev=0.1,
    #                         seed=SEED))
    conv2_weights = tf.get_variable("C2", shape=[ CONV_2_H, CONV_1_D, CONV_2_D],
                                    initializer=xavier_init2(CONV_2_H,  CONV_1_D, CONV_2_D))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[CONV_2_D]))


    # # conv3_weights = tf.Variable(
    # #    tf.truncated_normal([CONV_3_W, CONV_3_H, CONV_2_D, CONV_3_D],
    # #                        stddev=0.1,
    # #                        seed=SEED))
    conv3_weights = tf.get_variable("C3", shape=[CONV_3_H, CONV_2_D, CONV_3_D],
                                    initializer=xavier_init2(CONV_3_H,  CONV_2_D, CONV_3_D))
    conv3_biases = tf.Variable(tf.constant(0.1, shape=[CONV_3_D]))

    if CONV_1_PADDING == 'VALID':
        # WH = (NUM_ROWS - CONV_1_W + 2) / (POOL_1 * POOL_2 ) * (DATA_SIZE - CONV_1_H + 1) / (1 * 1) * CONV_2_D ################################################################
        WH=128*3 ################################################### todo !!!!
    else:
        WH = (NUM_ROWS) / (POOL_1 * POOL_2) * (DATA_SIZE) / (1 * 1) * CONV_2_D

    print('WH')
    print(WH)
    # fc1_weights = tf.Variable(  # fully connected, depth 512.
    #     tf.truncated_normal([WH, FULL_D],
    #                         stddev=0.1,
    #                         seed=SEED))
    fc1_weights = tf.get_variable("W1", shape=[WH, FULL_D], initializer=xavier_init(WH, FULL_D))

    fc1_biases = tf.Variable(tf.constant(0.1, shape=[FULL_D]))


    # fc11_weights = tf.Variable(  # fully connected, depth 512.
    #     tf.truncated_normal([512, 512],
    #                         stddev=0.1,
    #                         seed=SEED))
    # fc11_weights = tf.get_variable("W11", shape=[512, 512], initializer=xavier_init(512, 512))
    # fc11_biases = tf.Variable(tf.constant(0.1, shape=[512]))


    # fc2_weights = tf.Variable(
    #     tf.truncated_normal([FULL_D, NUM_LABELS],
    #                         stddev=0.1,
    #                         seed=SEED))
    fc2_weights = tf.get_variable("W2", shape=[FULL_D, NUM_LABELS], initializer=xavier_init(FULL_D, NUM_LABELS))


    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))


    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    def model(data, train):
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        # conv = tf.nn.conv2d(data,
        #                     conv1_weights,
        #                     strides=[1, 1, 1, 1],
        #                     padding=CONV_1_PADDING)
        conv = tf.nn.conv1d(data,conv1_weights, stride=1,padding="SAME")################################################################################   must change

        print('data', data)
        print('conv1_weights', conv1_weights)
        print('conv', conv)
        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        # relu = tf.nn.dropout(relu,d_rate)
        print(relu)
        #relu = tf.nn.dropout(relu, 0.5)
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.

        if True:
            # pool = tf.nn.max_pool(relu,
            #                   ksize=[1, POOL_1_k ],
            #                   strides=[1, POOL_1],
            #                   padding='SAME')
            pool = tf.layers.max_pooling1d(relu,POOL_1,strides=POOL_1, padding='VALID')
            print(pool)
        else:
            pool = relu
            print(pool)

        # conv = tf.nn.conv2d(pool,
        #                     conv2_weights,
        #                     strides=[1, 1, 1, 1],
        #                     padding='SAME')
        conv = tf.nn.conv1d(pool, conv2_weights, stride=1, padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
        # relu = tf.nn.dropout(relu,d_rate)

        if True:
            # pool = tf.nn.max_pool(relu,
            #                   ksize=[1, POOL_2_k],
            #                   strides=[1, POOL_2],
            #                   padding='SAME')
            pool = tf.layers.max_pooling1d(relu, POOL_2, strides=POOL_2,padding='VALID')
            print(pool)
        else:
            pool = relu
            print(pool)


        # conv = tf.nn.conv2d(pool,
        #                    conv3_weights,
        #                    strides=[1, 1, 1, 1],
        #                    padding='SAME')
        print('data', numpy.shape(data))
        conv = tf.nn.conv1d(pool, conv3_weights, stride=1, padding="SAME")
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv3_biases))
        # relu = tf.nn.dropout(relu, d_rate)
        if True:
            # pool = tf.nn.max_pool(relu,
            #                  ksize=[1, POOL_3_k],
            #                  strides=[1, POOL_3],
            #                  padding='SAME')
            pool = tf.layers.max_pooling1d(relu, POOL_3, strides=POOL_3,padding='VALID')
        else:
            pool = relu
            print(pool)


        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool.get_shape().as_list()
        print(pool_shape)
        reshape = tf.reshape(
            pool,
            [pool_shape[0], pool_shape[1] * pool_shape[2] ])

        print(reshape)
        print(fc1_weights)
        print(fc1_biases)
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        #hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        print("hidden", hidden)
        # print fc11
        # hidden = tf.nn.relu(tf.matmul(hidden, fc11_weights) + fc11_biases)


        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        print('before train work')
        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
            print('train work')
        return tf.matmul(hidden, fc2_weights) + fc2_biases

    # Training computation: logits + cross-entropy loss.
    logits = model(train_data_node, True)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=train_labels_node))

    # L2 regularization for the fully connected parameters. + tf.nn.l2_loss(fc11_weights) + tf.nn.l2_loss(fc11_biases) +

    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    # Add the regularization term to the loss.
    loss += 5e-4 * regularizers
    # loss += 0

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0)
    saver = tf.train.Saver(tf.all_variables())
    ##cross_entropy = -tf.reduce_sum(0)
    ##tf.scalar_summary("cross_entropy", cross_entropy)
    ##summary_op = tf.merge_all_summaries()

    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        0.0001,                # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        train_size,          # Decay step.
        0.95,                # Decay rate.
        staircase=True)
    # Use simple momentum for the optimization.
    optimizer = tf.train.AdamOptimizer(learning_rate,
                                           0.9).minimize(loss,
                                                         global_step=batch)
    #filter_summary = tf.image_summary("s", conv1_weights)

    # Predictions for the minibatch, validation set and test set.
    train_prediction = tf.nn.softmax(logits)
    # We'll compute them only once in a while by calling their {eval()} method.
    validation_prediction = tf.nn.softmax(model(validation_data_node,False))

    # Create a local session to run this computation.
    with tf.Session() as s:
        # Run all the initializers to prepare the trainable parameters.
        tf.initialize_all_variables().run()
        print('Initialized!')

        matrix_test = [[0 for col in range(len(STATES))] for row in range(len(STATES))]
        ##summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
        ##                                    graph_def=s.graph_def)
        # Loop through training steps.
        for step in range(int(num_epochs * train_size / BATCH_SIZE)):
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            #  batch_data = train_data[offset:(offset + BATCH_SIZE), :, :, :]
            batch_data = train_data[offset:(offset + BATCH_SIZE), :, :]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph is should be fed to.
            feed_dict = {train_data_node: batch_data,
                         train_labels_node: batch_labels}
            # Run the graph and fetch some of the nodes.
            _, l, lr, predictions = s.run(
                [optimizer, loss, learning_rate, train_prediction],
                feed_dict=feed_dict)
            # predictions = s.run(train_prediction,
            #     feed_dict=feed_dict)
            print('optimizer')
            print(_)
            print('loss')
            print(l)
            print('learning_rate')
            print(lr)

            if step % 100 == 0:
                print('Epoch %.2f' % (float(step) * BATCH_SIZE / train_size))
                print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                print('Minibatch error: %.1f%%' % error_rate(predictions,
                                                             batch_labels))
                print('Validation error: %.1f%%' % error_rate(
                    validation_prediction.eval(), validation_labels))
                #print 'logits: ', train_prediction.eval()
                sys.stdout.flush()
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                ###saver.save(s, checkpoint_path, global_step=step)
                ##if summary_op is not None:
                ##    summary_str = s.run(summary_op)
                ##    summary_writer.add_summary(summary_str, step)
                #summary_writer.add_summary(filter_summary, step)
        # Finally print the result!
        #start_time = time.time()
        #test_prediction.eval(), test_labels
        #print "--- %s seconds ---" % (time.time() -start_time)
        of = open(res_file_name, 'a')
        for i in range(16):
            test_data_node = tf.constant(test_data_list[i])
            test_labels = test_labels_list[i]
            test_prediction = tf.nn.softmax(model(test_data_node, False))
            # test_error = error_rate(test_prediction.eval(), test_labels)
            test_error, matrix_test \
                = error_draw_matrix_rate(test_prediction.eval(), test_labels, matrix_test)

            of.write('Test false alarm: %.1f%%\n' % false_alarm(
                test_prediction.eval(), test_labels))
            of.write('Test correct detection: %.1f%%\n' % correct_detection(
                test_prediction.eval(), test_labels))
            of.write('Test error: %.1f%%\n' % test_error)

            for i in range(len(STATES)):
                of.write('%s\n,' % matrix_test[i])

            of.flush()
            print('Test false alarm: %.1f%%' % false_alarm(
                test_prediction.eval(), test_labels))
            print('Test correct detection: %.1f%%' % correct_detection(
                test_prediction.eval(), test_labels))
            print('Test error: %.1f%%' % test_error)
            # saver.save(s, "w_0328.ckpt")
            if FLAGS.self_test:
                print('test_error', test_error)
                assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % (
                    test_error,)

        # test_data_node = tf.constant(test_data_list[i])
        # test_labels = test_labels_list[i]
        # test_prediction = tf.nn.softmax(model(test_data_node, False))
        # test_error = error_rate(test_prediction.eval(), test_labels)
        # of.write('Test Result for ID : %d\n' % (i + 1))
        # of.write('Test false alarm: %.1f%%\n' % false_alarm(
        #     test_prediction.eval(), test_labels))
        # of.write('Test correct detection: %.1f%%\n' % correct_detection(
        #     test_prediction.eval(), test_labels))
        # of.write('Test error: %.1f%%\n' % test_error)
        # of.flush()
        # print 'Test Result for ID : %d' % (i + 1)
        # print 'Test false alarm: %.1f%%' % false_alarm(
        #     test_prediction.eval(), test_labels)
        # print 'Test correct detection: %.1f%%' % correct_detection(
        #     test_prediction.eval(), test_labels)
        # print 'Test error: %.1f%%' % test_error
        # saver.save(s, 'w_0322')
        # if FLAGS.self_test:
        #     print 'test_error', test_error
        #     assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % (
        #         test_error,)
        #


        of.close()

if __name__ == '__main__':
    tf.app.run()
