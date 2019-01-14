import argparse
import extractData_hdh
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from fcn_model import Model

training_epochs = 200
batch_size = 500

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

    trainX, testX, trainY, testY = train_test_split(
        beacon_split_table, target_split_table, test_size=0.15, shuffle=False)
    print("train:", np.shape(trainX), "test:", np.shape(testX))
    # initialize

    sess = tf.Session()
    m1 = Model(sess, "m1", 0.0001)

    kf = KFold(n_splits=5)
    for enum, (train_index, test_index) in enumerate(kf.split(trainX)):
        print("KFold:", enum)
        print(train_index, test_index)
        trainX_kf = trainX[train_index]
        trainY_kf = trainY[train_index]
        validX_kf = trainX[test_index]
        validY_kf = trainY[test_index]

        sess.run(tf.global_variables_initializer())

        # train my model
        for epoch in range(training_epochs):
            # Validate
            if epoch%10 == 0:
                valid_average = 0
                valid_total_batch = int(validX_kf.shape[0] / batch_size)
                for valid_index in range(valid_total_batch):
                    valid_xs = validX_kf[valid_index*batch_size:(valid_index+1)*batch_size, :, :, :]
                    valid_ys = validY_kf[valid_index*batch_size:(valid_index+1)*batch_size, :]
                    accuracy, _, _, _= m1.get_accuracy(valid_xs, valid_ys)
                    valid_average += accuracy
                valid_average /= valid_total_batch
                print("Epoch:{}/{} Valid Accuracy:{}".format(epoch, training_epochs, valid_average))

            # Train
            avg_cost = 0
            total_batch = int(testY.shape[0] / batch_size)
            for i in range(total_batch):
                batch_xs = trainX_kf[i*batch_size:(i+1)*batch_size, :, :, :]
                batch_ys = trainY_kf[i*batch_size:(i+1)*batch_size, :]
                _, c, _, accuracy = m1.train(batch_xs, batch_ys)
                avg_cost += c / total_batch
            accuracy, _, _, _= m1.get_accuracy(trainX_kf, trainY_kf)
            print("Epoch:{}/{} Train Accuracy:{}".format(epoch, training_epochs, accuracy))

        print('Train average accuracy:', accuracy)
        print('Valid average accuracy:', valid_average)
        print('Test Accuracy:', m1.get_accuracy(testX[:, :, :, :], testY[:, :])[0])

    # test model and check accuracy

if __name__ == "__main__":
    main()