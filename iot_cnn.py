import argparse
import extractData_hdh
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

training_epochs = 50000
batch_size = 1000

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
            self.X_img = tf.placeholder(tf.float32, [None, 4, 24, 1])
            self.Y = tf.placeholder(tf.float32, [None, 14])
            
            with tf.name_scope("convolution1"):
                W1 = tf.get_variable("W1", shape=[2, 2, 1, 20])
                L1 = tf.nn.conv2d(self.X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
                L1 = tf.contrib.layers.batch_norm(L1, center=True, scale=True, is_training=phase, scope='bn1')
                L1 = tf.nn.relu(L1)
                L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)
                #4 24 20
                self.W1_hist = tf.summary.histogram("weights1", W1)

            with tf.name_scope("convolution2"):
                W2 = tf.get_variable("W2", shape=[2, 2, 20, 100])
                L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
                L3 = tf.contrib.layers.batch_norm(L2, center=True, scale=True, is_training=phase, scope='bn2')
                L2 = tf.nn.relu(L2)
                L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1], padding='SAME')
                L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)
                #2 12 100
                self.W2_hist = tf.summary.histogram("weights2", W2)

            with tf.name_scope("convolution3"):
                W3 = tf.get_variable("W3", shape=[2, 2, 100, 200])
                L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
                L3 = tf.contrib.layers.batch_norm(L3, center=True, scale=True, is_training=phase, scope='bn3')
                L3 = tf.nn.relu(L3)
                L3 = tf.nn.dropout(L3, keep_prob=self.keep_prob)
                #2 12 200
                L3_flat = tf.reshape(L3, [-1, 200 * 2 * 12])
                #200*2*12
                self.W3_hist = tf.summary.histogram("weights3", W3)

            with tf.name_scope("fully_connected1"):
                FC_W1 = tf.get_variable("FC_W1", shape=[200 * 2 * 12, 400],
                                     initializer=tf.contrib.layers.xavier_initializer())
                FC_b1 = tf.get_variable("FC_b1", shape=[400])
                FC_L1 = tf.nn.relu(tf.matmul(L3_flat, FC_W1) + FC_b1)
                FC_L1 = tf.nn.dropout(FC_L1, keep_prob=self.keep_prob)
                # 400
                self.FC_W1_hist = tf.summary.histogram("weights_FC1", FC_W1)
                self.FC_b1_hist = tf.summary.histogram("bias_FC1", FC_b1)

            with tf.name_scope("fully_connected2"):
                # Final FC 400 inputs -> 14 outputs
                FC_W2 = tf.get_variable("FC_W2", shape=[400, 14],
                                     initializer=tf.contrib.layers.xavier_initializer())
                FC_b2 = tf.get_variable("FC_b2", shape=[14])
                self.logits = tf.matmul(FC_L1, FC_W2) + FC_b2
                self.FC_W2_hist = tf.summary.histogram("weights_FC2", FC_W2)
                self.FC_b2_hist = tf.summary.histogram("bias_FC2", FC_b2)

        # define cost/loss & optimizer
        with tf.name_scope("cost"):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.Y))  
            self.cost_summ = tf.summary.scalar("cost", self.cost)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.accuracy_sum = tf.summary.scalar("accuracy", self.accuracy)
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

    beacon_split_table = data.expand_time_onehot_beacon_table(beacon_table, 4)[:, :, 1:, np.newaxis]
    target_split_table = data.expand_time_onehot_beacon_table(target_table, 4)[:, -1, 1:]

    # shuffle dataset
    idx = np.random.permutation(len(beacon_split_table))
    shuffled_beacon_table, shuffled_target_table = beacon_split_table[idx], target_split_table[idx]

    # split to train, valid, test dataset
    beacon_train, beacon_test, target_train, target_test = train_test_split(shuffled_beacon_table, shuffled_target_table, test_size=0.3)
    beacon_valid, beacon_test, target_valid, target_test = train_test_split(beacon_test, target_test, test_size=0.5)


    print(beacon_train.shape)
    print(beacon_test.shape)
    print(target_train.shape)
    print(target_test.shape)

    # initialize
    sess = tf.Session()

    m1 = Model(sess, "m1", 0.00001)
    writer = tf.summary.FileWriter("./logs/iot_r0_02")
    writer.add_graph(m1.sess.graph)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    print('Learning Started!')
    global_step = 0
    # train my model
    for epoch in range(training_epochs):
        if epoch%10 == 0:
            print('Valid Accuracy:', m1.get_accuracy(beacon_valid[:, :, :, :], target_valid[:, :]))
            #ckpt_path = saver.save(sess, "saved/train0", epoch)
        avg_cost = 0
        total_batch = int(target_test.shape[0] / batch_size)

        for i in range(total_batch):
            batch_xs = beacon_train[i*batch_size:(i+1)*batch_size, :, :, :]
            batch_ys = target_train[i*batch_size:(i+1)*batch_size, :]
            summary, c, _, accuracy = m1.train(batch_xs, batch_ys)
            #writer.add_summary(summary, global_step=global_step)
            avg_cost += c / total_batch
            global_step += 1

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}, accuracy:{}'.format(avg_cost, accuracy))
    print('Learning Finished!')

    # test model and check accuracy
    print('Accuracy:', m1.get_accuracy(beacon_test[:, :, :, :], target_test[:, :]))

if __name__ == "__main__":
    main()
