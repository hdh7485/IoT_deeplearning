import argparse
import extractData_hdh
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os
import matplotlib

<<<<<<< HEAD
=======
training_epochs = 50000
batch_size = 100

>>>>>>> f3ab411b94fdbec72a44fd8016002ad32fdeacee
tf.set_random_seed(777)  # reproducibility

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

# train Parameters
seq_length = 5
hidden_dim = 10
output_dim = 14
learning_rate = 0.01
<<<<<<< HEAD
iterations = 1000
=======
iterations = 500
>>>>>>> f3ab411b94fdbec72a44fd8016002ad32fdeacee

parser = argparse.ArgumentParser()
parser.add_argument("--data_directory", default="../raw_data/")
args = parser.parse_args()

data = extractData_hdh.IOTDataset()
data.load_json_files(args.data_directory)
beacon_table = data.make_time_onehot_beacon_table()[:, 1:]
target_table = data.make_time_onehot_target_table()[:, 1:]

# shuffle dataset
#idx = np.random.permutation(len(beacon_split_table))
#shuffled_beacon_table, shuffled_target_table = beacon_split_table[idx], target_split_table[idx]

# split to train, valid, test dataset
beacon_train, beacon_test, target_train, target_test = train_test_split(beacon_table, target_table, test_size=0.3)
beacon_valid, beacon_test, target_valid, target_test = train_test_split(beacon_test, target_test, test_size=0.5)

# build datasets
def build_dataset(time_series, y_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_length):
        _x = time_series[i:i + seq_length, :]
        #_y = time_series[i + seq_length, [-1]]  # Next close price
        _y = y_series[i+seq_length, :]
        #print(_x, "->", _y)
        dataX.append(_x)
        dataY.append(_y)
    return np.array(dataX), np.array(dataY)

trainX, trainY = build_dataset(beacon_train, target_train, seq_length)
testX, testY = build_dataset(beacon_test, target_test, seq_length)
<<<<<<< HEAD
validX, validY = build_dataset(beacon_valid, target_valid, seq_length)
print(validX.shape)
print(validY.shape)
=======
print(trainX.shape)
print(trainY.shape)
>>>>>>> f3ab411b94fdbec72a44fd8016002ad32fdeacee

# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, 24])
Y = tf.placeholder(tf.float32, [None, 14])

# build a LSTM network
cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# RMSE
<<<<<<< HEAD
targets = tf.placeholder(tf.float32, [None, 14])
predictions = tf.placeholder(tf.float32, [None, 14])
=======
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
>>>>>>> f3ab411b94fdbec72a44fd8016002ad32fdeacee
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(iterations):
<<<<<<< HEAD
        if i%10 == 0:
            _, step_loss = sess.run([train, loss], feed_dict={X: validX, Y: validY})
            print("[valid: {}] loss: {}".format(i, step_loss))

        _, step_loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
=======
        _, step_loss = sess.run([train, loss], feed_dict={
                                X: trainX, Y: trainY})
>>>>>>> f3ab411b94fdbec72a44fd8016002ad32fdeacee
        print("[step: {}] loss: {}".format(i, step_loss))

    # Test step
    test_predict = sess.run(Y_pred, feed_dict={X: testX})
    rmse_val = sess.run(rmse, feed_dict={
                    targets: testY, predictions: test_predict})
    print("RMSE: {}".format(rmse_val))