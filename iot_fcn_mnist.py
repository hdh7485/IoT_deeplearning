import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from mnist_model import Model
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_x = mnist.train.images
train_y = mnist.train.labels
test_x = mnist.test.images
test_y = mnist.test.labels

training_epochs = 500
batch_size = 100

def main():
    # initialize
    sess = tf.Session()

    m1 = Model(sess, "m1", 0.0001)

    kf = KFold(n_splits=5)
    for enum, (train_index, test_index) in enumerate(kf.split(train_x)):
        print("KFold:", enum)
        print(train_index, test_index)
        print(np.shape(train_x))

        beacon_train = train_x[train_index]
        target_train = train_y[train_index]
        beacon_valid = train_x[test_index]
        target_valid = train_y[test_index]

        sess.run(tf.global_variables_initializer())

        global_step = 0
        # train my model
        for epoch in range(training_epochs):
            # Validate
            if epoch%10 == 0:
                valid_average = 0
                valid_total_batch = int(target_valid.shape[0] / batch_size)
                for valid_index in range(valid_total_batch):
                    valid_xs = beacon_valid[valid_index*batch_size:(valid_index+1)*batch_size, :]
                    valid_ys = target_valid[valid_index*batch_size:(valid_index+1)*batch_size, :]
                    accuracy, out_X, out_Y, out_Y_pre = m1.get_accuracy(valid_xs, valid_ys)
                    valid_average += accuracy
                valid_average /= valid_total_batch

            # Train
            avg_cost = 0
            total_batch = int(train_x.shape[0] / batch_size)
            for i in range(total_batch):
                batch_xs = beacon_train[i*batch_size:(i+1)*batch_size, :]
                batch_ys = target_train[i*batch_size:(i+1)*batch_size, :]
                _, c, _, accuracy = m1.train(batch_xs, batch_ys)
                avg_cost += c / total_batch
                global_step += 1

        print('Train average accuracy:', accuracy)
        print('Valid average accuracy:', valid_average)
        print('Test Accuracy:', m1.get_accuracy(test_x[:, :], test_y[:, :])[0])

    # test model and check accuracy

if __name__ == "__main__":
    main()
