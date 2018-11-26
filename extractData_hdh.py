import csv
from datetime import datetime
import numpy as np
import os
import argparse
import json
import pprint
import pickle

class IOTDataset:
    def __init__(self):
        self.feature_names_list = [
          "millisecond",
          "rssi",
          "beacon_loc_cd",
          "wifi_rssi",
          "wifi_rssi_2",
          "wifi_rssi_3",
          "wifi_rssi_4",
          "doctor_id",
          "gyroscope_field_x",
          "gyroscope_field_y",
          "gyroscope_field_z",
          "accelerometer_x",
          "accelerometer_y",
          "accelerometer_z",
          "magnetic_field_z",
          "magnetic_field_y",
          "magnetic_field_x",
          "accuracy",
          "txPower",
          "act_type",
          "real_loc_cd",
          "b_reg_date"
          ]
        self.feature_data_list = []
        self.target_time_list = []
        self.target_names_list = []
        self.target_list = []
        self.first_line = True

    def make_beacon_name_list(self, beacon_list):
        self.beacon_name_list = []
        for beacon in beacon_list:
            if beacon not in self.beacon_name_list:
                self.beacon_name_list.append(beacon)
        self.beacon_name_list.sort()

    def make_target_name_list(self, target_list):
        self.target_name_list = []
        for target in target_list:
            if target not in self.target_name_list:
                self.target_name_list.append(target)
        self.target_name_list.sort()

    def load_json_files(self, raw_data_directory='../raw_data', use_saved_data=True):
        saved_data_directory = os.path.join(raw_data_directory, 'raw_data.pickle')
        if use_saved_data and os.path.isfile(saved_data_directory):
            print('start load the saved json data')
            with open(saved_data_directory, 'rb') as f:
                pickle_data = pickle.load(f)
            self.json_time_beacon_rssi = pickle_data[0]
            self.json_target_list = pickle_data[1]
            self.json_target = pickle_data[2]

        else:
            print('No saved json data')
            self.date_paths = []
            dates = os.listdir(raw_data_directory)
            for date in dates:
                if int(date) >= 20170403:
                    self.date_paths.append(os.path.join(raw_data_directory, date))
        
            self.json_path_list = []
            for path in self.date_paths:
                for file in os.listdir(path):
                    if file.endswith("log_treat_json"):
                        self.json_path_list.append(os.path.join(path, file))

            self.json_files = []
            for path in self.json_path_list:
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.endswith(".txt"):
                            self.json_files.append(os.path.join(root, file))

            self.json_time_beacon_rssi = []
            self.json_target_list = []
            for json_file in self.json_files:
                with open(json_file, 'r') as f:
                    loaded_json_data = json.load(f)
                f.close()

                for beacon_data in loaded_json_data.values():
                    for a_data in beacon_data:
                        for b_data in a_data.values():
                            for c_data in b_data:
                                self.json_time_beacon_rssi.append([c_data['millisecond'], c_data['beacon_loc_cd'], c_data['rssi']])
                                self.json_target_list.append([c_data['millisecond'], c_data['real_loc_cd']])

            self.json_time_beacon_rssi.sort(key=lambda x: x[0])
            self.json_target_list.sort(key=lambda x: x[0])
            self.json_target = np.array(self.json_target_list)

            with open(saved_data_directory, 'wb') as f:
                pickle.dump([self.json_time_beacon_rssi, self.json_target_list, self.json_target], f)
        print('finish load josn files')

    def make_time_onehot_beacon_table(self, pickle_data_directory='../raw_data/onehot_beacon_table.pickle', use_saved_data=True):
        self.make_beacon_name_list(np.array(self.json_time_beacon_rssi)[:, 1])

        if use_saved_data and os.path.isfile(pickle_data_directory):
            print('loading the saved pickle data')
            with open(pickle_data_directory, 'rb') as f:
                pickle_data = pickle.load(f)
            self.json_time_rssi_table_list = pickle_data[0]
            self.json_time_rssi_table = pickle_data[1]

        else:
            print('No saved pickle data')
            self.json_time_rssi_table_list = [] 
            for data in self.json_time_beacon_rssi:
                row = list(np.zeros(len(self.beacon_name_list)+1))
                row[0] = data[0]
                row[self.beacon_name_list.index(data[1])+1] = data[2]
                self.json_time_rssi_table_list.append(row)
            self.json_time_rssi_table = np.array(self.json_time_rssi_table_list)

            with open(pickle_data_directory, 'wb') as f:
                pickle.dump([self.json_time_rssi_table_list, self.json_time_rssi_table], f)
        print('finish make beacon table')
        return self.json_time_rssi_table
<<<<<<< HEAD
=======
   
    def expand_time_onehot_beacon_table(self, raw_table, split_rows):
        s0,s1 = raw_table.strides
        m,n = raw_table.shape
        return np.lib.stride_tricks.as_strided(raw_table, shape=(m-split_rows+1, split_rows, n), strides=(s0, s0, s1))
>>>>>>> add_tensorboard

    def make_time_onehot_target_table(self, pickle_data_directory='../raw_data/onehot_target_table.pickle', use_saved_data=True):
        self.make_target_name_list(np.array(self.json_target_list)[:, 1])
        if use_saved_data and os.path.isfile(pickle_data_directory):
            print('loading the saved pickle data')
            with open(pickle_data_directory, 'rb') as f:
                pickle_data = pickle.load(f)
            self.json_target_table_list = pickle_data[0]
            self.json_target_table = pickle_data[1]
<<<<<<< HEAD

=======
>>>>>>> add_tensorboard
        else:
            print('No saved pickle data')
            self.json_target_table_list = []
            for data in self.json_target_list:
                row = list(np.zeros(len(self.target_name_list)+1))
                row[0] = data[0]
                row[self.target_name_list.index(data[1])+1] = 1
                self.json_target_table_list.append(row)
            self.json_target_table = np.array(self.json_target_table_list)

            with open(pickle_data_directory, 'wb') as f:
                pickle.dump([self.json_target_table_list, self.json_target_table], f)
        print('finish make target table')
        return self.json_target_table

    def load_csv_files(self, raw_data_directory):
        self.date_paths = []
        dates = os.listdir(raw_data_directory)
        for date in dates:
            if int(date) >= 20170403:
                self.date_paths.append(os.path.join(raw_data_directory, date))
        #print(self.date_paths)
        
        self.csv_path_list = []
        for path in self.date_paths:
            for file in os.listdir(path):
                if file.endswith("log_treat_csv"):
                    self.csv_path_list.append(os.path.join(path, file))

        self.csv_files = []
        for path in self.csv_path_list:
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(".csv"):
                        self.csv_files.append(os.path.join(root, file))
        #print(self.csv_files)

    def read_feature(self):
        f = open('./data/beacon_data_170201to170327.csv', 'r')
        csv_reader = csv.reader(f)
        first_line = True
        for line in csv_reader:
            if first_line:
                first_line = False
                continue
            self.feature_data_list.append(line) 
        f.close()
        self.feature_names = np.array(self.feature_names_list)
        self.feature_data = np.array(self.feature_data_list)

    def read_target(self):
        f = open('./data/write_data_170201to170327.csv', 'r')
        csv_reader = csv.reader(f)
        first_line = True
        for line in csv_reader:
            if first_line:
                first_line = False
                continue
            if len(line[-1]) < 2:
                continue
        
            if not line[-1] in self.target_names_list:
                self.target_names_list.append(line[-1])
            self.target_list.append(line[-1])
            self.target_time_list.append(line[-2])
        self.target_names_list.sort()
        self.target_list = [self.target_names_list.index(x) for x in self.target_list]
        f.close()
        self.target_time = np.array(self.target_time_list)
        self.target_names = np.array(self.target_names_list)
        self.target = np.array(self.target_list)

    def load_cnn_format(self):
        self.cnn_feature_data_list = []
        target_index = 0
        cnn_bundle = []
        for line in self.feature_data_list:
            if self.str_to_datetime(line[3]) <\
            self.str_to_datetime(self.target_time_list[target_index]):
                cnn_bundle.append(line)
            else:
                self.cnn_feature_data_list.append(cnn_bundle)
                cnn_bundle = []
                target_index += 1
                if target_index > len(self.target_time_list) - 1:
                    break

        self.cnn_feature_data = np.array(self.cnn_feature_data_list)
        #return self.cnn_feature_data

    def str_to_datetime(self, time_string):
        return datetime.strptime(time_string, '%Y-%m-%d %H:%M:%S')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_directory", default="../raw_data/")
    args = parser.parse_args()

    data = IOTDataset()

    data.load_json_files(args.data_directory)
    beacon_table = data.make_time_onehot_beacon_table()
    target_table = data.make_time_onehot_target_table()
    expand_target_table = data.expand_time_onehot_beacon_table(beacon_table, 4)[:, :, 1:]

    print(beacon_table.shape)
    print(target_table.shape)
    print(expand_target_table.shape)
    print(expand_target_table)

if __name__ == "__main__":
    main()
