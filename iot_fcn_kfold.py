import argparse
import extractData_hdh
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

training_epochs = 500
batch_size = 100

class Model:
    def __init__(self, sess, name, learning_rate):
        self.sess = sess
        self.name = name
        self.learning_rate = learning_rate
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing
            self.keep_prob = tf.placeholder(tf.float32)
            self.phase= tf.placeholder(tf.bool)
            self.X_img = tf.placeholder(tf.float32, [None, 4, 24, 1])
            self.X = tf.reshape(self.X_img, [-1, 4 * 24])
            self.negative_X = tf.negative(self.X)
            self.Y = tf.placeholder(tf.float32, [None, 14])
            
            with tf.name_scope("fully_connected1"):
                FC_W1 = tf.get_variable("FC_W1", shape=[4 * 24, 8],
                                     initializer=tf.contrib.layers.xavier_initializer())
                FC_b1 = tf.get_variable("FC_b1", shape=[8])
                FC_L1 = tf.matmul(self.negative_X, FC_W1) + FC_b1
                FC_L1 = tf.contrib.layers.batch_norm(FC_L1, center=True, scale=True, is_training=self.phase, scope='FC_bn1')
                FC_L1 = tf.nn.relu(FC_L1)
                FC_L1 = tf.nn.dropout(FC_L1, keep_prob=self.keep_prob)
                # 400
                self.FC_W1_hist = tf.summary.histogram("weights_FC1", FC_W1)
                self.FC_b1_hist = tf.summary.histogram("bias_FC1", FC_b1)

            with tf.name_scope("fully_connected3"):
                # Final FC 400 inputs -> 14 outputs
                FC_W3 = tf.get_variable("FC_W3", shape=[8, 14],
                                     initializer=tf.contrib.layers.xavier_initializer())
                FC_b3 = tf.get_variable("FC_b3", shape=[14])
                self.logits = tf.matmul(FC_L1, FC_W3) + FC_b3
                self.FC_W3_hist = tf.summary.histogram("weights_FC3", FC_W3)
                self.FC_b3_hist = tf.summary.histogram("bias_FC3", FC_b3)

        # define cost/loss & optimizer
        with tf.name_scope("cost"):
            self.softmax = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits, labels=self.Y)
            self.cost = tf.reduce_mean(self.softmax) 
            self.cost_summ = tf.summary.scalar("cost", self.cost)

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.cost)
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.accuracy_sum = tf.summary.scalar("accuracy", self.accuracy)
        self.merged_summary = tf.summary.merge_all()


    def predict(self, x_test, keep_prop=1.0):
        return self.sess.run(self.logits, feed_dict={self.X_img: x_test, self.keep_prob: keep_prop, self.phase:False})

    def get_accuracy(self, x_test, y_test, keep_prop=1.0):
        return self.sess.run([self.accuracy, self.negative_X, self.Y, self.logits], feed_dict={
            self.X_img: x_test, self.Y: y_test, self.keep_prob: keep_prop, self.phase:True})

    def train(self, x_data, y_data, keep_prop=0.7):
        return self.sess.run([self.merged_summary, self.cost, self.optimizer, self.accuracy], feed_dict={
            self.X_img: x_data, self.Y: y_data, self.keep_prob: keep_prop, self.phase:True})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_directory", default="../raw_data/")
    args = parser.parse_args()

    data = extractData_hdh.IOTDataset()
    data.load_json_files(args.data_directory)
    beacon_table = data.make_time_onehot_beacon_table().astype(np.float)
    target_table = data.make_time_onehot_target_table().astype(np.float)

    beacon_split_table = data.expand_time_onehot_beacon_table(beacon_table, 4)[:, :, 1:, np.newaxis]
    target_split_table = data.expand_time_onehot_beacon_table(target_table, 4)[:, -1, 1:]

    beacon_train, beacon_test, target_train, target_test = train_test_split(
        beacon_split_table, target_split_table, test_size=0.15, shuffle=False)
    print("train:", np.shape(beacon_train), "test:", np.shape(beacon_test))
    # initialize
    sess = tf.Session()

    m1 = Model(sess, "m1", 0.0001)
    writer = tf.summary.FileWriter("./logs/iot_r0_02")
    writer.add_graph(m1.sess.graph)
    saver = tf.train.Saver()

    kf = KFold(n_splits=5)
    for enum, (train_index, test_index) in enumerate(kf.split(beacon_split_table)):
        print("KFold:", enum)
        print(train_index, test_index)
        beacon_train = beacon_split_table[train_index]
        target_train = target_split_table[train_index]
        beacon_valid = beacon_split_table[test_index]
        target_valid = target_split_table[test_index]

        sess.run(tf.global_variables_initializer())

        global_step = 0
        # train my model
        for epoch in range(training_epochs):
            # Validate
            if epoch%10 == 0:
                valid_average = 0
                valid_total_batch = int(target_valid.shape[0] / batch_size)
                for valid_index in range(valid_total_batch):
                    valid_xs = beacon_valid[valid_index*batch_size:(valid_index+1)*batch_size, :, :, :]
                    valid_ys = target_valid[valid_index*batch_size:(valid_index+1)*batch_size, :]
                    accuracy, out_X, out_Y, out_Y_pre = m1.get_accuracy(valid_xs, valid_ys)
                    valid_average += accuracy
                valid_average /= valid_total_batch

            # Train
            avg_cost = 0
            total_batch = int(target_test.shape[0] / batch_size)
            for i in range(total_batch):
                batch_xs = beacon_train[i*batch_size:(i+1)*batch_size, :, :, :]
                batch_ys = target_train[i*batch_size:(i+1)*batch_size, :]
                summary, c, _, accuracy = m1.train(batch_xs, batch_ys)
                avg_cost += c / total_batch
                global_step += 1

        print('Train average accuracy:', accuracy)
        print('Valid average accuracy:', valid_average)
        print('Test Accuracy:', m1.get_accuracy(beacon_test[:, :, :, :], target_test[:, :])[0])

    # test model and check accuracy

if __name__ == "__main__":
    main()
