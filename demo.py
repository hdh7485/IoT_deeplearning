import argparse
import extractData_hdh
import tensorflow as tf
import numpy as np

learning_rate = 0.0001
training_epochs = 500
batch_size = 5

class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing
            self.keep_prob = tf.placeholder(tf.float32)
            self.X_img = tf.placeholder(tf.float32, [None, 5, 24])
            self.X_img = tf.reshape(self.X_img, [-1, 5, 24, 1])
            self.Y = tf.placeholder(tf.float32, [None, 14])
            
            with tf.name_scope("convolution1"):
                W1 = tf.Variable(tf.random_normal([2, 2, 1, 10], stddev=0.01))
                L1 = tf.nn.conv2d(self.X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
                L1 = tf.nn.relu(L1)
                L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)
                #L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                #5 24 5
                self.W1_hist = tf.summary.histogram("weights1", W1)
                self.convolution1_hist = tf.summary.histogram("convolution1", W1)

            with tf.name_scope("convolution2"):
                W2 = tf.Variable(tf.random_normal([2, 2, 10, 20], stddev=0.01))
                L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
                L2 = tf.nn.relu(L2)
                L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1], padding='SAME')
                L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)
                # L2 ImgIn shape=(?, 3, 12, 20)
                L2_flat = tf.reshape(L2, [-1, 20 * 3 * 12])

                self.W2_hist = tf.summary.histogram("weights2", W2)
                self.convolution2_hist = tf.summary.histogram("convolution2", W2)

            with tf.name_scope("convolution3"):
                W3 = tf.get_variable("W4", shape=[20 * 3 * 12, 200],
                                     initializer=tf.contrib.layers.xavier_initializer())
                b3 = tf.Variable(tf.random_normal([200]))
                L3 = tf.nn.relu(tf.matmul(L2_flat, W3) + b3)
                L3 = tf.nn.dropout(L3, keep_prob=self.keep_prob)

                W3_hist = tf.summary.histogram("weights3", W3)
                convolution3_hist = tf.summary.histogram("convolution3", W3)

            with tf.name_scope("fully_connected"):
                # L5 Final FC 400 inputs -> 14 outputs
                FC_W = tf.get_variable("W5", shape=[200, 14],
                                     initializer=tf.contrib.layers.xavier_initializer())
                FC_b = tf.Variable(tf.random_normal([14]))
                self.logits = tf.matmul(L3, FC_W) + FC_b

                self.FC_W_hist = tf.summary.histogram("weights1", FC_W)
                self.FC_b_hist = tf.summary.histogram("bias1", FC_b)
                self.FC_layer_hist = tf.summary.histogram("fully_connected", FC_W)

        # define cost/loss & optimizer
        with tf.name_scope("cost"):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.Y))  
            self.cost_summ = tf.summary.scalar("cost", self.cost)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.accuracy_summ = tf.summary.scalar("accuracy", self.accuracy)
        self.merged_summary = tf.summary.merge_all()

    def predict(self, x_test, keep_prop=1.0):
        return self.sess.run(self.logits, feed_dict={self.X_img: x_test, self.keep_prob: keep_prop})

    def get_accuracy(self, x_test, y_test, keep_prop=1.0):
        return self.sess.run(self.accuracy, feed_dict={self.X_img: x_test, self.Y: y_test, self.keep_prob: keep_prop})

    def train(self, x_data, y_data, keep_prop=0.7):
        return self.sess.run([self.merged_summary, self.cost, self.optimizer, self.accuracy], feed_dict={
            self.X_img: x_data, self.Y: y_data, self.keep_prob: keep_prop})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_directory", default="../raw_data/")
    args = parser.parse_args()

    data = extractData_hdh.IOTDataset()
    data.load_json_files(args.data_directory)
    beacon_table = data.make_time_onehot_beacon_table()
    target_table = data.make_time_onehot_target_table()

    print(beacon_table)
    #(237017, 25)
    print(target_table)
    #(237017, 15)

    beacon_split_table = beacon_table[0:237000, 1:].reshape(47400, 5, 24)
    beacon_train, beacon_test = beacon_split_table[:47300, :, :, np.newaxis], beacon_split_table[47300:, :, :, np.newaxis]
    print(beacon_test[0])
    target_split_table = target_table[0:237000, 1:].reshape(47400, 5, 14)
    target_train, target_test = target_split_table[:47300, -1, :], target_split_table[47300:, -1, :]

    print(beacon_train.shape)
    print(beacon_test.shape)
    print(target_train.shape)
    print(target_test.shape)

    # initialize
    sess = tf.Session()
    
    writer = tf.summary.FileWriter("./logs/iot_r0_01")
    writer.add_graph(sess.graph)

    m1 = Model(sess, "m1")

    sess.run(tf.global_variables_initializer())

    print('Learning Started!')
    global_step = 0
    # train my model
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(target_test.shape[0] / batch_size)

        for i in range(total_batch):
            batch_xs = beacon_train[i*batch_size:(i+1)*batch_size, :, :, :]
            batch_ys = target_train[i*batch_size:(i+1)*batch_size, :]
            summary, c, _, accuracy = m1.train(batch_xs, batch_ys)
            writer.add_summary(summary, global_step=global_step)
            avg_cost += c / total_batch
            global_step += 1

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}, accuracy:{}'.format(avg_cost, accuracy))

    print('Learning Finished!')

    # test model and check accuracy
    print('Accuracy:', m1.get_accuracy(beacon_test[:, :, :, :], target_test[:, :]))


if __name__ == "__main__":
    main()
