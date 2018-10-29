import csv

class IOTDataset:
    def __init__(self):
        self.feature_names = ['loc_cd', 'rssi', 'doctor_id', 'reg_date', 'magnetic_field_x', 'magnetic_field_y', 'magnetic_field_z', 'wifi1', 'wifi2', 'wifi3', 'wifi4', 'index']
        self.feature_data = []
        self.target_time = []
        self.target_names = []
        self.target = []
        self.first_line = True

        self.read_feature()
        self.read_target()

    def read_feature(self):
        f = open('./data/beacon_data_170201to170327.csv', 'r')
        csv_reader = csv.reader(f)
        first_line = True
        for line in csv_reader:
            if first_line:
                first_line = False
                continue
            self.feature_data.append(line) 
        f.close()

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
        
            if not line[-1] in self.target_names:
                self.target_names.append(line[-1])
            self.target.append(line[-1])
            self.target_time.append(line[-2])
        self.target_names.sort()
        self.target = [self.target_names.index(x) for x in self.target]
        f.close()

def main():
    data = IOTDataset()
    print(data.target_names)
    print(data.target)

if __name__ == "__main__":
    main()
