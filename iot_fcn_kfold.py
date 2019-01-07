import argparse
import extractData_hdh
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from fcn_model import Model

training_epochs = 500
batch_size = 100

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
