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

def make_cnn_data:
    pass
     
def main(argv=None):  # pylint: disable=unused-argument
    if FLAGS.self_test:
        print('Running self-test.')
        train_data, train_labels = fake_data(256)
        validation_data, validation_labels = fake_data(16)
        test_data, test_labels = fake_data(256)
        num_epochs = 1
    else:

        validation_data, validation_labels = extractData.extract_data_oned(numRows = NUM_ROWS, numData = VALIDATION_SIZE, states = STATES, moves=MOVES, doctors=DOCTORS, dates=DATES,labels = LABELS, mode = 'validate',DATA_SIZE = DATA_SIZE,NUM_CHANNELS=NUM_CHANNELS, ONED=True)
        validation_data = validation_data[:, :, 0, :]
        print("validation_data", numpy.shape(validation_data))

        #train_data, train_labels = extractData.extract_data(numRows = NUM_ROWS, numData = 10000, mode = 'train')
        #test_data, test_labels = extractData.extract_data(numRows = NUM_ROWS, numData = 2000, mode = 'test')
        #validation_data, validation_labels = extractData.extract_data(numRows = NUM_ROWS, numData = 2000, mode = 'validate')
        # Generate a validation set.
        num_epochs = NUM_EPOCHS


    if gfile.Exists(FLAGS.train_dir):
        gfile.DeleteRecursively(FLAGS.train_dir)
    gfile.MakeDirs(FLAGS.train_dir)

    train_size = train_labels.shape[0]

    train_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, NUM_ROWS,  NUM_CHANNELS))
    print("train_data_node", train_data_node)
    print("train_data_node", train_data_node[0])
    train_labels_node = tf.placeholder(tf.float32,
                                       shape=(BATCH_SIZE, NUM_LABELS))

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

        of.close()

if __name__ == '__main__':
    tf.app.run()
