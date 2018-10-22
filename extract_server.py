import csv
import shutil
import os
import numpy
import sys
import operator
import collections
import pickle
import math
#from sets import Set
import datetime
from datetime import timedelta
from datetime import date
import MySQLdb
import pylab
import time

import MySQLdb.cursors
import numpy as np


HOST_IP = "183.96.198.135"
HOST_PORT = 13400
USER_ID = "htsm"
PASS_WD = "htsm12#$"
DB_NAME = "el_crm_visit"

db = MySQLdb.connect(host=HOST_IP,
                    user=USER_ID,
                    passwd=PASS_WD,
                    port=HOST_PORT,
                    charset='utf8',
                    cursorclass=MySQLdb.cursors.DictCursor,
                    db=DB_NAME)

cur = db.cursor(MySQLdb.cursors.DictCursor)


# user configuration

BASE = '/home/mskim/tensorflow/test/data_file'

MIN_ROWS = 32
#MAX_VAL = 200.0

#DATA_SIZE = 20
# DATA_SIZE = 10
NUM_LABELS = 0
NUM_CHANNELS=0
#
DOCTORS = ['00000000', '00000049', '00000063']
BEACONS = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'O1', 'O2','P1', 'P2', 'T1',  'Z1', 'Z6', 'Z7', 'Z8', 'Z9']
CHAIRS = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'O1', 'O2','P1', 'P2', 'T1' ]
TIME_DELTA =  30
States = 'C1,C2,C3,C4,C5,C6,C7,C8,O1,O2,P1,P2,T1'
Labels = {}

X_width = 27
X_height = 10
Y_width =18
Y_height = 1
B_max=66
B_min=4
m_x_max=1180
m_x_min=-11360
m_y_max=615
m_y_min=-14690
m_z_max=296
m_z_min=-2261
g_x_max=12
g_x_min=-11
g_y_max=7
g_y_min=-5
g_z_max=6
g_z_min=-5
a_x_max=11
a_x_min=-22
a_y_max=17
a_y_min=-38
a_z_max=39
a_z_min=-36
num = 27 #to erase x_data[0] = [000000000000]
flag =0



def input_data():
    for DID in DOCTORS:
        command = "select tbli.pk_idx, " \
                  "tbli.fk_idx, " \
                  "tbli.fd_write_date, " \
                  "tbli.fd_loc_info, tbli.fd_loc_cd as beacon_loc, " \
                  "tbli.fd_doctor_id, tbli.fd_rssi as rssi," \
                  " tbli.fd_accelerometer_x as ax, \
                    tbli.fd_accelerometer_y as ay,\
                    tbli.fd_accelerometer_z as az,\
                    tbli.fd_gyroscope_field_x as gx,\
                    tbli.fd_gyroscope_field_y as gy,\
                    tbli.fd_gyroscope_field_z as gz,\
                    tbli.fd_magnetic_field_x as mx,\
                    tbli.fd_magnetic_field_y as my,\
                    tbli.fd_magnetic_field_z as mz,\
                    tbli.fd_reg_date as time\
                    from tbl_beacon_loc_info as tbli\
                    where\
                    tbli.fd_doctor_id = '" + DID + "' and tbli.fd_reg_date >= DATE_ADD(NOW(), INTERVAL '-50' SECOND)\
                    and tbli.fd_reg_date <= DATE_ADD(NOW(), INTERVAL 0 SECOND)\
                    order by tbli.fd_reg_date; "
        # command = "select tbli.pk_idx, " \
        #           "tbli.fk_idx, " \
        #           "tbli.fd_write_date, " \
        #           "tbli.fd_loc_info, tbli.fd_loc_cd as beacon_loc, " \
        #           "tbli.fd_doctor_id, tbli.fd_rssi as rssi," \
        #           " tbli.fd_accelerometer_x as ax, \
        #             tbli.fd_accelerometer_y as ay,\
        #             tbli.fd_accelerometer_z as az,\
        #             tbli.fd_gyroscope_field_x as gx,\
        #             tbli.fd_gyroscope_field_y as gy,\
        #             tbli.fd_gyroscope_field_z as gz,\
        #             tbli.fd_magnetic_field_x as mx,\
        #             tbli.fd_magnetic_field_y as my,\
        #             tbli.fd_magnetic_field_z as mz,\
        #             tbli.fd_reg_date as time\
        #             from tbl_beacon_loc_info as tbli\
        #             where\
        #             tbli.fd_doctor_id = '00000000' and tbli.fd_reg_date >= DATE_ADD('2016-12-20 13:33:59', INTERVAL - 1 DAY)\
        #     and tbli.fd_reg_date <= DATE_ADD('2016-12-25 13:33:59', INTERVAL '-1 -1' DAY_MINUTE)\
        #     order by tbli.fd_reg_date ;"

        cur.execute(command)
        row2 = cur.fetchall()

        if row2 !="":
            break

    return row2

def save_to_server(did,loc):
    str_date = datetime.date.today()
    str_date = str_date.strftime("%y%m%d")

    cur.execute("""INSERT INTO tbl_doctor_loc_estimation_info
    (fd_date, fd_doctor_id, fd_loc_cd, fd_reg_date) VALUES (%s, %s, %s, NOW());""",
                (str_date, did, loc))
    db.commit()



def make_x(row, x_data):

    p_x_date = ""
    row_count=0
    count=0
    temp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    temp = np.array(temp)

    p_time=""
    for i in range(len(row)):
        if p_x_date != row[i]['time']:
            row_count=row_count+1
    if row_count >26 :
        for i in range(len(row)):
            if p_time == row[i]['time'] : #in same time
                for j in range(0,19):
                    if BEACONS[j] == str(row[i]['beacon_loc']):
                        temp[j] = float(int(row[i]['rssi'])  + 110 - B_min) / float(B_max - B_min) #insert rssi
                        break
            else: #not in same time
                if i!= 0:
                    temp[18] = float(float(row[i-1]['mx']) - m_x_min) / float(m_x_max - m_x_min)
                    temp[19] = float(float(row[i-1]['my']) - m_y_min) / float(m_y_max - m_y_min)
                    temp[20] = float(float(row[i-1]['mz']) - m_z_min) / float(m_z_max - m_z_min)
                    temp[21] = float(float(row[i-1]['gx']) - g_x_min) / float(g_x_max - g_x_min)
                    temp[22] = float(float(row[i-1]['gy']) - g_y_min) / float(g_y_max - g_y_min)
                    temp[23] = float(float(row[i-1]['gz']) - g_z_min) / float(g_z_max - g_z_min)
                    temp[24] = float(float(row[i-1]['ax']) - a_x_min) / float(a_x_max - a_x_min)
                    temp[25] = float(float(row[i-1]['ay'])- a_y_min) / float(a_y_max - a_y_min)
                    temp[26] = float(float(row[i-1]['az']) - a_z_min) / float(a_z_max - a_z_min)
                x_data = np.append(x_data, temp)
                count = count +1
                if count >num:  #size
                    break
                temp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                for j in range(0, 19):
                    if BEACONS[j] == str(row[i]['beacon_loc']):
                        temp[j] = float(int(row[i]['rssi'])  + 110 - B_min) / float(B_max - B_min)  # insert rssi
                        break

            p_time = row[i]['time']

        # x_data = np.reshape(x_data, (-1, X_width))
        x_data = np.reshape(x_data, (-1,1, X_width))
        x_data = np.delete(x_data, 0,0)
    print(numpy.shape(x_data))
    return x_data

def extract_data_oned(states=None, labels=None, groups=None, personIDs=None, numRows=None, numData=None, mode=None, NUM_CHANNELS=None,
                     ONED=True, DATA_SIZE=None):

    results = input_data()
    x_data = numpy.array([])
    x_data = make_x(results, x_data)
    did = str(results[0]['fd_doctor_id'])

    if ONED:
        data = numpy.ndarray(shape=(numData, numRows, 1, NUM_CHANNELS), dtype=numpy.float32)
    else:
        data = numpy.ndarray(shape=(numData, numRows, DATA_SIZE, 1), dtype=numpy.float32)
    # res_idx =  len(results) - 1
    # max_row_idx = results[res_idx].shape[0] - numRows
    # row_idx = max_row_idx
    if ONED:
        # data[:, :, 0, :] = results[res_idx][row_idx:row_idx + numRows, 0:NUM_CHANNELS]
        data[ :,:, :, :] = x_data[1][:,:]

    print(numpy.shape(data))

    return data, did


    if __name__ == '__main__':
        data, dataLabel = extract_data(numRows = 300, numData = 1000)

        dataLabel[numDataCount, filelabels[res_idx]] = 1.0

        numDataCount = numDataCount + 1
    return data
