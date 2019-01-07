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
                FC_W1 = tf.get_variable("FC_W1", shape=[4 * 24, 16],
                                     initializer=tf.contrib.layers.xavier_initializer())
                FC_b1 = tf.get_variable("FC_b1", shape=[16])
                FC_L1 = tf.matmul(self.negative_X, FC_W1) + FC_b1
                FC_L1 = tf.contrib.layers.batch_norm(FC_L1, center=True, scale=True, is_training=self.phase, scope='FC_bn1')
                FC_L1 = tf.nn.relu(FC_L1)
                FC_L1 = tf.nn.dropout(FC_L1, keep_prob=self.keep_prob)
                # 400
                self.FC_W1_hist = tf.summary.histogram("weights_FC1", FC_W1)
                self.FC_b1_hist = tf.summary.histogram("bias_FC1", FC_b1)

            with tf.name_scope("fully_connected3"):
                # Final FC 400 inputs -> 14 outputs
                FC_W3 = tf.get_variable("FC_W3", shape=[16, 14],
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
class Model2:
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
            self.negative_X = tf.negative(self.X_img)
            self.Y = tf.placeholder(tf.float32, [None, 14])
            
            with tf.name_scope("convolution1"):
                W1 = tf.get_variable("W1", shape=[2, 2, 1, 8])
                L1 = tf.nn.conv2d(self.negative_X, W1, strides=[1, 1, 1, 1], padding='SAME')
                L1 = tf.contrib.layers.batch_norm(L1, center=True, scale=True, is_training=self.phase, scope='bn1')
                L1 = tf.nn.relu(L1)
                L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)
                L1_flat = tf.reshape(L1, [-1, 8 * 4 * 24])
                #4 24 8
                #self.W1_hist = tf.summary.histogram("weights1", W1)

            with tf.name_scope("fully_connected1"):
                FC_W1 = tf.get_variable("FC_W1", shape=[8 * 4 * 24, 64],
                                     initializer=tf.contrib.layers.xavier_initializer())
                FC_b1 = tf.get_variable("FC_b1", shape=[64])
                FC_L1 = tf.matmul(L1_flat, FC_W1) + FC_b1
                FC_L1 = tf.contrib.layers.batch_norm(FC_L1, center=True, scale=True, is_training=self.phase, scope='FC_bn1')
                FC_L1 = tf.nn.relu(FC_L1)
                FC_L1 = tf.nn.dropout(FC_L1, keep_prob=self.keep_prob)
                # 400
                self.FC_W1_hist = tf.summary.histogram("weights_FC1", FC_W1)
                self.FC_b1_hist = tf.summary.histogram("bias_FC1", FC_b1)

            with tf.name_scope("fully_connected2"):
                FC_W2 = tf.get_variable("FC_W2", shape=[64, 64],
                                     initializer=tf.contrib.layers.xavier_initializer())
                FC_b2 = tf.get_variable("FC_b2", shape=[64])
                FC_L2 = tf.matmul(FC_L1, FC_W2) + FC_b2
                FC_L2 = tf.contrib.layers.batch_norm(FC_L2, center=True, scale=True, is_training=self.phase, scope='FC_bn2')
                FC_L2 = tf.nn.relu(FC_L2)
                FC_L2 = tf.nn.dropout(FC_L2, keep_prob=self.keep_prob)
                # 400
                self.FC_W1_hist = tf.summary.histogram("weights_FC2", FC_W2)
                self.FC_b1_hist = tf.summary.histogram("bias_FC2", FC_b2)

            with tf.name_scope("fully_connected3"):
                # Final FC 400 inputs -> 14 outputs
                FC_W3 = tf.get_variable("FC_W3", shape=[64, 14],
                                     initializer=tf.contrib.layers.xavier_initializer())
                FC_b3 = tf.get_variable("FC_b3", shape=[14])
                self.logits = tf.matmul(FC_L2, FC_W3) + FC_b3
                self.FC_W3_hist = tf.summary.histogram("weights_FC3", FC_W2)
                self.FC_b3_hist = tf.summary.histogram("bias_FC3", FC_b2)

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
        return self.sess.run([self.accuracy, self.negative_X, self.Y, self.logits], feed_dict={self.X_img: x_test, self.Y: y_test, self.keep_prob: keep_prop, self.phase:True})

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
    number_of_target = target_split_table.astype(np.int).sum(axis=0)
    print(number_of_target)
    #plt.plot(number_of_target)
    #plt.show()

    # shuffle dataset
    idx = np.random.permutation(len(beacon_split_table))
    shuffled_beacon_table, shuffled_target_table \
    = beacon_split_table[idx], target_split_table[idx]
    
    kf = KFold(n_splits=5)
    print(kf.split(beacon_split_table, target_split_table))

    # split to train, valid, test dataset
    #beacon_train, beacon_test, target_train, target_test = train_test_split(
    #    shuffled_beacon_table, shuffled_target_table, test_size=0.3, shuffle=False)
    beacon_train, beacon_test, target_train, target_test = train_test_split(
        beacon_split_table, target_split_table, test_size=0.3, shuffle=False)
    beacon_valid, beacon_test, target_valid, target_test = train_test_split(
        beacon_test, target_test, test_size=0.5, shuffle=False)
    # check imported data
    '''
    for k in range(100):
        fig = plt.figure()
        plt.subplot(211)
        X = np.negative(np.reshape(beacon_valid[-k*4], (4, 24)))
        print(X)
        plt.imshow(X, cmap="gray")
        plt.subplot(212)
        Y = target_valid[-k*4]
        print(Y)
        plt.imshow(np.expand_dims(Y, axis=0), cmap="gray")
        #plt.savefig('img/valid_first_{}.png'.format(k), bbox_inches='tight')
        plt.show()    
    for k in range(10):
        fig = plt.figure()
        plt.subplot(211)
        X = np.negative(np.reshape(beacon_valid[-k], (4, 24)))
        print(X)
        plt.imshow(X, cmap="gray")
        plt.subplot(212)
        Y = target_valid[-k]
        print(Y)
        plt.imshow(np.expand_dims(Y, axis=0), cmap="gray")
        plt.savefig('img/valid_last_{}.png'.format(k), bbox_inches='tight')
        #plt.show()    

    for k in range(10):
        fig = plt.figure()
        plt.subplot(211)
        X = np.negative(np.reshape(beacon_train[k], (4, 24)))
        print(X)
        plt.imshow(X, cmap="gray")
        plt.subplot(212)
        Y = target_train[k]
        print(Y)
        plt.imshow(np.expand_dims(Y, axis=0), cmap="gray")
        plt.savefig('img/train_first_{}.png'.format(k), bbox_inches='tight')
        #plt.show()    
    '''
    '''
    for k in range(100):
        fig = plt.figure()
        plt.subplot(211)
        X = np.negative(np.reshape(beacon_test[-k*4], (4, 24)))
        print(X)
        plt.imshow(X, cmap="gray")
        plt.subplot(212)
        Y = target_test[-k*4]
        print(Y)
        plt.imshow(np.expand_dims(Y, axis=0), cmap="gray")
        #plt.savefig('img/train_last_{}.png'.format(k), bbox_inches='tight')
        plt.show()    
    '''

    # initialize
    sess = tf.Session()

    m1 = Model(sess, "m1", 0.0001)
    writer = tf.summary.FileWriter("./logs/iot_r0_02")
    writer.add_graph(m1.sess.graph)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    print('Learning Started!')
    global_step = 0
    # train my model
    for epoch in range(training_epochs):
        if epoch%10 == 0:
            valid_average = 0
            valid_total_batch = int(target_valid.shape[0] / batch_size)
            for valid_index in range(valid_total_batch):
                valid_xs = beacon_valid[valid_index*batch_size:(valid_index+1)*batch_size, :, :, :]
                valid_ys = target_valid[valid_index*batch_size:(valid_index+1)*batch_size, :]
                accuracy, out_X, out_Y, out_Y_pre = m1.get_accuracy(valid_xs, valid_ys)
                #print('X:{}\nY:{}\nY_pre:{}\nValid Accuracy:{}'.format(
                #    out_X[0], out_Y[0], out_Y_pre[0], accuracy))
                #print('Valid Accuracy:{}'.format(accuracy))
                valid_average += accuracy
                #ckpt_path = saver.save(sess, "saved/train0", epoch)
            valid_average /= valid_total_batch
            print('Valid average accuracy:{}'.format(valid_average))

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
