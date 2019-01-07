import tensorflow as tf

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