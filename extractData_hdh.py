import csv
from datetime import datetime
import numpy as np

class IOTDataset:
    def __init__(self):
        self.feature_names_list = ['loc_cd', 'rssi', 'doctor_id', 'reg_date', 'magnetic_field_x', 'magnetic_field_y', 'magnetic_field_z', 'wifi1', 'wifi2', 'wifi3', 'wifi4', 'index']
        self.feature_data_list = []

        self.target_time_list = []
        self.target_names_list = []
        self.target_list = []

        self.first_line = True

        self.read_feature()
        self.read_target()
        self.load_cnn_format()

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
    data = IOTDataset()
    print(data.cnn_feature_data)

if __name__ == "__main__":
    main()
