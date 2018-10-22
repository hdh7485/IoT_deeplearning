import gzip
import os
import sys
import urllib

import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy
import tensorflow as tf
import extractData
import numpy as np
import pandas as pd

# States = ['C1','C2','C3','C4','C5','C6','C7','C8','O1','O2','P1','P2','T1','Z1','A1', 'A2']
Path = "/home/mskim/IoT/raw_data/data_D/"
makeDate = '0603'
X_width = 47
X_height = 10
Y_width =12
Y_height = 1
num = 27360 #228*10
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
W_max = 81
W_min = 16
States = ['C1','C2','C3','C4','C5','C6','C7','C8','O1','O2','P1','P2','T1','Z1','A1', 'A2']
# States = ['C1','C2','C3','C4','C5','C6','C7','C8','O1','O2','P1','P2','T1','Z1','A1', 'A2','MOVE']
BEACONS = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'O1', 'O2', 'P1', 'P2', 'T1',
           'Z1', 'Z2', 'Z5', 'Z6', 'Z7', 'Z8', 'Z9', 'A1', 'A2', 'DC1', 'DC2', 'DC3', 'DC4', 'DC5', 'DC6', 'DC7', 'DC8', 'DO2', 'DP1', 'DP2', 'DT1']
MOVE = [['C1_C2', 'C1_C3', 'C1_C4', 'C1_C5', 'C1_C6', 'C1_C7', 'C1_C8', 'C1_O1', 'C1_O2', 'C1_P1', 'C1_P2', 'C1_T1', 'C1_Z1', 'C1_A1', 'C1_A2'],
        ['C2_C1', 'C2_C3', 'C2_C4', 'C2_C5', 'C2_C6', 'C2_C7', 'C2_C8', 'C2_O1', 'C2_O2', 'C2_P1', 'C2_P2', 'C2_T1', 'C2_Z1', 'C2_A1', 'C2_A2'],
        ['C3_C1', 'C3_C2', 'C3_C4', 'C3_C5', 'C3_C6', 'C3_C7', 'C3_C8', 'C3_O1', 'C3_O2', 'C3_P1', 'C3_P2', 'C3_T1', 'C3_Z1', 'C3_A1', 'C3_A2'],
        ['C4_C1', 'C4_C2', 'C4_C3', 'C4_C5', 'C4_C6', 'C4_C7', 'C4_C8', 'C4_O1', 'C4_O2', 'C4_P1', 'C4_P2', 'C4_T1', 'C4_Z1', 'C4_A1', 'C4_A2'],
        ['C5_C1', 'C5_C2', 'C5_C3', 'C5_C4', 'C5_C6', 'C5_C7', 'C5_C8', 'C5_O1', 'C5_O2', 'C5_P1', 'C5_P2', 'C5_T1', 'C5_Z1', 'C5_A1', 'C5_A2'],
        ['C6_C1', 'C6_C2', 'C6_C3', 'C6_C4', 'C6_C5', 'C6_C7', 'C6_C8', 'C6_O1', 'C6_O2', 'C6_P1', 'C6_P2', 'C6_T1', 'C6_Z1', 'C6_A1', 'C6_A2'],
        ['C7_C1', 'C7_C2', 'C7_C3', 'C7_C4', 'C7_C5', 'C7_C6', 'C7_C8', 'C7_O1', 'C7_O2', 'C7_P1', 'C7_P2', 'C7_T1', 'C7_Z1', 'C7_A1', 'C7_A2'],
        ['C8_C1', 'C8_C2', 'C8_C3', 'C8_C4', 'C8_C5', 'C8_C6', 'C8_C7', 'C8_O1', 'C8_O2', 'C8_P1', 'C8_P2', 'C8_T1', 'C8_Z1', 'C8_A1', 'C8_A2'],
        ['O1_C1', 'O1_C2', 'O1_C3', 'O1_C4', 'O1_C5', 'O1_C6', 'O1_C7', 'O1_C8', 'O1_O2', 'O1_P1', 'O1_P2', 'O1_T1', 'O1_Z1', 'O1_A1', 'O1_A2'],
        ['O2_C1', 'O2_C2', 'O2_C3', 'O2_C4', 'O2_C5', 'O2_C6', 'O2_C7', 'O2_C8', 'O2_O1', 'O2_P1', 'O2_P2', 'O2_T1', 'O2_Z1', 'O2_A1', 'O2_A2'],
        ['P1_C1', 'P1_C2', 'P1_C3', 'P1_C4', 'P1_C5', 'P1_C6', 'P1_C7', 'P1_C8', 'P1_O1', 'P1_O2', 'P1_P2', 'P1_T1', 'P1_Z1', 'P1_A1', 'P1_A2'],
        ['P2_C1', 'P2_C2', 'P2_C3', 'P2_C4', 'P2_C5', 'P2_C6', 'P2_C7', 'P2_C8', 'P2_O1', 'P2_O2', 'P2_P1', 'P2_T1', 'P2_Z1', 'P2_A1', 'P2_A2'],
        ['T1_C1', 'T1_C2', 'T1_C3', 'T1_C4', 'T1_C5', 'T1_C6', 'T1_C7', 'T1_C8', 'T1_O1', 'T1_O2', 'T1_P1', 'T1_P2', 'T1_Z1', 'T1_A1', 'T1_A2'],
        ['Z1_C1', 'Z1_C2', 'Z1_C3', 'Z1_C4', 'Z1_C5', 'Z1_C6', 'Z1_C7', 'Z1_C8', 'Z1_O1', 'Z1_O2', 'Z1_P1', 'Z1_P2', 'Z1_T1', 'Z1_A1', 'Z1_A2'],
        ['A1_C1', 'A1_C2', 'A1_C3', 'A1_C4', 'A1_C5', 'A1_C6', 'A1_C7', 'A1_C8', 'A1_O1', 'A1_O2', 'A1_P1', 'A1_P2', 'A1_T1', 'A1_Z1', 'A1_A2'],
        ['A2_C1', 'A2_C2', 'A2_C3', 'A2_C4', 'A2_C5', 'A2_C6', 'A2_C7', 'A2_C8', 'A2_O1', 'A2_O2', 'A2_P1', 'A2_P2', 'A2_T1', 'A2_Z1', 'A2_A1']
        ]
Dates = ['20170331','20170403','20170404','20170405','20170406','20170407','20170410','20170411','20170412','20170413','20170414',
         '20170417','20170419','20170420','20170421','20170424','20170425','20170426','20170427','20170428']
BASE = '/home/mskim/IoT/raw_data/raw_data'
# Path1 = '00000004_treat_csv'
# Path2 = '00000049_treat_csv'
Doctor = ['00000004','00000049']
CP_Path = '/home/mskim/IoT/raw_data/cp_files'
# Merge_Path1 = '/home/mskim/IoT/raw_data/merge_data_file_0_magneto'
Merge_Path = '/home/mskim/IoT/raw_data/new_data'
mode=['_treat_csv','_move_csv']
filenames=[]
filelabels=[]
firsttime=0



def make_x(table, x_data, y_data):
    date = table["Date"]
    dposition = table["d_position"]
    beconId = table["beconId"]
    rssi = table["rssi"]
    w1 = table["W1"]
    w2 = table["W2"]
    w3 = table["W3"]
    w4 = table["W4"]

    magnetic_x = table["m_x"]
    magnetic_y = table["m_y"]
    magnetic_z = table["m_z"]
    gyroscope_x = table["g_x"]
    gyroscope_y = table["g_y"]
    gyroscope_z = table["g_z"]
    accelerometer_x = table["a_x"]
    accelerometer_y = table["a_y"]
    accelerometer_z = table["a_z"]

    p_x_date = 0
    count=0
    pre_x = [0 for row in range(X_width)]

    temp = [0 for row in range(X_width)]

    temp1 = [0 for row in range(len(States))]
    # pre_x=np.array(pre_x)
    temp = np.array(temp)
    temp1 = np.array(temp1)
    for n, i in enumerate(date):
        if n != 0 :
            if i == p_x_date:
                for j in range(0,34):
                    if BEACONS[j] == beconId[n]:
                        temp[j] = float(rssi[n] + 110 - B_min) / float(B_max - B_min)

                p_x_date = i
            else:
                for i in range(0,38):
                    if temp[i]==0 or numpy.isnan(temp[i]) :
                        temp[i]=pre_x[i]
                    else:
                        pre_x[i]=float(temp[i])
                y_data = np.append(y_data, temp1)
                x_data = np.append(x_data, temp)
                temp =  [0 for row in range(X_width)]

                temp1 = [0 for row in range(len(States))]

                for j in range(0, 34):
                    # print(beconId[5])
                    # print(BEACONS[j])
                    if BEACONS[j]==beconId[n]:
                        temp[j] = float(rssi[n] + 110 - B_min) / float(B_max - B_min)

                temp[34] = float(float(w1[n-1]) + 100 - W_min) / float(W_max - W_min)
                temp[35] = float(float(w2[n-1]) + 100 - W_min) / float(W_max - W_min)
                temp[36] = float(float(w3[n-1]) + 100 - W_min) / float(W_max - W_min)
                temp[37] = float(float(w4[n-1]) + 100 - W_min) / float(W_max - W_min)
                temp[38] = float(accelerometer_x[n] - a_x_min) / float(a_x_max - a_x_min)
                temp[39] = float(accelerometer_y[n] - a_y_min) / float(a_y_max - a_y_min)
                temp[40] = float(accelerometer_z[n] - a_z_min) / float(a_z_max - a_z_min)
                temp[41] = float(gyroscope_x[n] - g_x_min) / float(g_x_max - g_x_min)
                temp[42] = float(gyroscope_y[n] - g_y_min) / float(g_y_max - g_y_min)
                temp[43] = float(gyroscope_z[n] - g_z_min) / float(g_z_max - g_z_min)
                temp[44] = float(magnetic_x[n] - m_x_min) / float(m_x_max - m_x_min)
                temp[45] = float(magnetic_y[n] - m_y_min) / float(m_y_max - m_y_min)
                temp[46] = float(magnetic_z[n] - m_z_min) / float(m_z_max - m_z_min)

                p_x_date = i

    x_data = np.reshape(x_data, (-1, X_width))
    x_data = np.delete(x_data, 0, 0)
    return x_data, y_data



def make_x_move(table, x_data, y_data):
    date = pd.Series.tolist(table["Date"])
    dposition = pd.Series.tolist(table["d_position"])
    beconId = pd.Series.tolist(table["beconId"])
    rssi = pd.Series.tolist(table["rssi"])
    w1 = pd.Series.tolist(table["W1"])
    w2 = pd.Series.tolist(table["W2"])
    w3 = pd.Series.tolist(table["W3"])
    w4 = pd.Series.tolist(table["W4"])

    magnetic_x = pd.Series.tolist(table["m_x"])
    magnetic_y = pd.Series.tolist(table["m_y"])
    magnetic_z = pd.Series.tolist(table["m_z"])
    gyroscope_x = pd.Series.tolist(table["g_x"])
    gyroscope_y = pd.Series.tolist(table["g_y"])
    gyroscope_z = pd.Series.tolist(table["g_z"])
    accelerometer_x = pd.Series.tolist(table["a_x"])
    accelerometer_y = pd.Series.tolist(table["a_y"])
    accelerometer_z =pd.Series.tolist(table["a_z"])

    p_x_date = 0
    count=0
    pre_x = [0 for row in range(X_width)]

    temp = [0 for row in range(X_width)]

    temp1 =  [0 for row in range(len(States))]
    # pre_x=np.array(pre_x)
    temp = np.array(temp)
    temp1 = np.array(temp1)
    for n, i in enumerate(date):
        if n != 0 :
            if i == p_x_date:
                for j in range(0,34):
                    if BEACONS[j] == beconId[n]:
                        temp[j] = float(rssi[n] + 110 - B_min) / float(B_max - B_min)

                p_x_date = i
            else:
                for h in range(0,38):
                    if temp[h]==0 or numpy.isnan(temp[h]) :
                        temp[h]=pre_x[h]
                    else:
                        pre_x[h]=float(temp[h])
                y_data = np.append(y_data, temp1)
                x_data = np.append(x_data, temp)
                temp =  [0 for row in range(X_width)]

                temp1 =  [0 for row in range(len(States))]

                for j in range(0, 34):
                    if BEACONS[j]==beconId[n]:
                        temp[j] = float(rssi[n] + 110 - B_min) / float(B_max - B_min)

                temp[34] = float(float(w1[n - 1]) + 100 - W_min) / float(W_max - W_min)
                temp[35] = float(float(w2[n - 1]) + 100 - W_min) / float(W_max - W_min)
                temp[36] = float(float(w3[n - 1]) + 100 - W_min) / float(W_max - W_min)
                temp[37] = float(float(w4[n - 1]) + 100 - W_min) / float(W_max - W_min)
                temp[38] = float(accelerometer_x[n] - a_x_min) / float(a_x_max - a_x_min)
                temp[39] = float(accelerometer_y[n] - a_y_min) / float(a_y_max - a_y_min)
                temp[40] = float(accelerometer_z[n] - a_z_min) / float(a_z_max - a_z_min)
                temp[41] = float(gyroscope_x[n] - g_x_min) / float(g_x_max - g_x_min)
                temp[42] = float(gyroscope_y[n] - g_y_min) / float(g_y_max - g_y_min)
                temp[43] = float(gyroscope_z[n] - g_z_min) / float(g_z_max - g_z_min)
                temp[44] = float(magnetic_x[n] - m_x_min) / float(m_x_max - m_x_min)
                temp[45] = float(magnetic_y[n] - m_y_min) / float(m_y_max - m_y_min)
                temp[46] = float(magnetic_z[n] - m_z_min) / float(m_z_max - m_z_min)

                p_x_date = i

    x_data = np.reshape(x_data, (-1, X_width))
    x_data = np.delete(x_data, 0, 0)
    return x_data, y_data


def devide(x_data, y_data, tr_x_data, tr_y_data):
    for i in range(0, len(x_data)):
        tr_x_data = np.append(tr_x_data,x_data[i])
        tr_y_data = np.append(tr_y_data, y_data[i])
    tr_x_data = np.reshape(tr_x_data, (-1, X_width))
    return x_data,y_data,tr_x_data,tr_y_data


def make_matrix(table,  x_data, y_data, tr_x_data, tr_y_data):
    x_data, y_data = make_x(table, x_data, y_data)
    x_data, y_data, tr_x_data, tr_y_data=devide(x_data, y_data, tr_x_data, tr_y_data)
    return tr_x_data, tr_y_data

def make_matrix_move(table,  x_data, y_data, tr_x_data, tr_y_data):
    x_data, y_data = make_x_move(table, x_data, y_data)
    x_data, y_data, tr_x_data, tr_y_data=devide(x_data, y_data, tr_x_data, tr_y_data)
    return tr_x_data, tr_y_data

def save(table, tr_x_data, tr_y_data, f, dirPath):
    table_name = str(table)
    if f==0:
        np.savetxt(dirPath+table_name+'_'+makeDate, tr_x_data,delimiter=',')
        print table_name
        print len(tr_x_data)
        tr_x_data=[]
        tr_y_data=[]
    else:
        table = table.split('_')
        table_name = 'm'+ table[0]
        np.savetxt(dirPath+table_name+'_'+makeDate, tr_x_data, delimiter=',')
        print table_name
        print len(tr_x_data)
        tr_x_data = []
        tr_y_data = []

    return tr_x_data, tr_y_data

def makedir_merge(filenames,f):
    CPName = CP_Path

    if f==0:#treat_mode
        for name in filenames:
            a = name.split('/')
            CPName1 = os.path.join(CPName, a[-2])
            if os.path.exists(CPName1) == False:
                os.system('mkdir ' + CPName1)  # ../C1
                # os.system('cat ' + name + " > " + Merge_Path + "/" + States[j] + "/" + States[j] + ".csv")
                # print 'cat ' + name + " > " + Merge_Path + "/" + States[j] + ".csv"
                # os.system('cp ' + name + " " + CP_Path + "/" + States[j] + "/" )
            doctorid = a[-3].split('_')
            CPName2 = os.path.join(CPName1, doctorid[0])
            if os.path.exists(CPName2) == False:
                os.system('mkdir ' + CPName2)  # ../C1/doctorid
            date = a[-4]
            CPName3 = os.path.join(CPName2, date)
            if os.path.exists(CPName3) == False:
                os.system('mkdir ' + CPName3)  # ../C1/doctorid/date
            os.system('cp ' + name + " " + CPName3 + "/")
            os.system('cat ' + CPName3 + '/*.csv > ' + CPName3 + '/' + a[-2] + ".csv")
            print 'cat ' + CPName3 + '/*.csv > ' + CPName3 + '/' + a[-2] + ".csv"
            print "found files %s" % (a[-2])

    else:#move_mode
        for name in filenames:
            a = name.split('/')
            CPName1 = os.path.join(CPName, a[-2])
            if os.path.exists(CPName1) == False:
                os.system('mkdir ' + CPName1)  # ../C1
                # os.system('cat ' + name + " > " + Merge_Path + "/" + States[j] + "/" + States[j] + ".csv")
                # print 'cat ' + name + " > " + Merge_Path + "/" + States[j] + ".csv"
                # os.system('cp ' + name + " " + CP_Path + "/" + States[j] + "/" )
            doctorid = a[-3].split('_')
            CPName2 = os.path.join(CPName1, doctorid[0])
            if os.path.exists(CPName2) == False:
                os.system('mkdir ' + CPName2)  # ../C1/doctorid
            date = a[-4]
            CPName3 = os.path.join(CPName2, date)
            if os.path.exists(CPName3) == False:
                os.system('mkdir ' + CPName3)  # ../C1/doctorid/date
            os.system('cp ' + name + " " + CPName3 + "/")
            os.system('cat ' + CPName3 + '/*.csv > ' + CPName3 + '/' + a[-2] + ".csv")
            print 'cat ' + CPName3 + '/*.csv > ' + CPName3 + '/' + a[-2] + ".csv"
            print "found files %s" % (a[-2])


def make_dir(dirName, Bid, Did, Date):
    Path = dirName

    if os.path.exists(Path) == False:
        os.system('mkdir ' + Path)  # ../C1
    Path2 = os.path.join(Path, Bid)
    if os.path.exists(Path2) == False:
        os.system('mkdir ' + Path2)  # ../C1
    Path3 = os.path.join(Path2, Did)
    if os.path.exists(Path3) == False:
        os.system('mkdir ' + Path3)  # ../C1
    Path4 = os.path.join(Path3, Date)
    if os.path.exists(Path4) == False:
        os.system('mkdir ' + Path4)  # ../C1







def read_all_csv(filenames,filelabels):
    dirName = BASE
    CPName = CP_Path
    # for i in range(len(States)):
    if os.path.exists(dirName):
        for fn in os.listdir(dirName):
            dirName1 = os.path.join(dirName,fn)
            for i in range(0,2):
                for f in range(0,2):
                    dirName2 = os.path.join(dirName1, Doctor[i]+mode[f])
                    if f==0 :   #treat mode
                        for j in range(len(States)):
                            dirName3 = os.path.join(dirName2, States[j])
                            if os.path.exists(dirName3):
                                if os.path.isdir(dirName3):
                                    for fn2 in os.listdir(dirName3):
                                        filenames.append(dirName3 + '/' +fn2)
                                makedir_merge(filenames,f)
                                filenames = []
                    else: #move mode
                        for j in range(len(MOVE)):
                            for h in range(len(MOVE[0])):
                                dirName3 = os.path.join(dirName2, MOVE[j][h])
                                if os.path.exists(dirName3):
                                    if os.path.isdir(dirName3):
                                        for fn2 in os.listdir(dirName3):
                                            filenames.append(dirName3 + '/' + fn2)
                                    makedir_merge(filenames,f)
                                    filenames = []
    return filenames, filelabels


def mergefiles():
    dirName = CP_Path

    if os.path.exists(dirName): #cp_files/
        for j in range(len(States)):
            dirName1 = os.path.join(dirName, States[j]) #merge_data_new/C1

            if os.path.exists(dirName):
                for fn in os.listdir(dirName):
                    dirName1 = os.path.join(dirName, fn)
                    for i in range(0, 2):
                        for f in range(0, 2):
                            dirName2 = os.path.join(dirName1, Doctor[i] + mode[f])
                            if f == 0:  # treat mode
                                for j in range(len(States)):
                                    dirName3 = os.path.join(dirName2, States[j])
                                    if os.path.isdir(dirName3):
                                        for fn2 in os.listdir(dirName3):
                                            filenames.append(dirName3 + '/' + fn2)

            for i in range(0, 2):
                dirName2 = os.path.join(dirName1, Doctor[i])
            if os.path.isdir(dirName1):
                os.system('cat '+dirName1+'/'+States[j]+'/*.csv > '  +dirName1+'/'+States[j]+'/'+ States[j] + ".csv")
                print 'cat '+dirName1+'/'+States[j]+'/*.csv > ' +dirName1+'/'+States[j]+'/'+ States[j] + ".csv"
                print "found files %s" % (States[j])
        for k in range(len(MOVE)):
            dirName1 = os.path.join(dirName, MOVE[k])  # merge_data_new/C1
            if os.path.isdir(dirName1):
                os.system('cat ' + dirName1 + '/' + MOVE[k] + '/*.csv > ' + dirName1 + '/' + MOVE[k] + '/' + MOVE[k] + ".csv")
                print 'cat ' + dirName1 + '/' + MOVE[k] + '/*.csv > ' + dirName1 + '/' + MOVE[k] + '/' + MOVE[k] + ".csv"
                print "found files %s" % (MOVE[k])
    return filenames, filelabels


filenames=[]
filelabels = []
filenames, filelabels = read_all_csv(filenames = filenames, filelabels = filelabels)
# mergefiles()

x_data=np.array([])
y_data=np.array([])
tr_x_data=np.array([])
tr_y_data=np.array([])

if os.path.exists(Merge_Path) == False:
    os.system('mkdir ' + Merge_Path)  #

if os.path.exists(CP_Path) == False:
    os.system('mkdir ' + CP_Path)  #


#
for i in range(len(States)):
    for j in range(len(Doctor)):
        for h in range(len(Dates)):
            dirName = CP_Path+'/'+States[i]+'/'+Doctor[j]+'/'+ Dates[h]+'/'
            if os.path.exists(dirName):
                    f= pd.read_table(dirName+'/'+States[i]+'.csv',sep=',',names=['Date','d_position','beconId','doctorId','rssi','txPower', 'accuracy','PHONE_beacon_id','a_x','a_y','a_z','g_x','g_y','g_z','m_x','m_y','m_z','CHECK','W1','W2','W3','W4','millis'])
                    if len(f) != 0:
                        tr_x_data, tr_y_data= make_matrix(f, x_data, y_data, tr_x_data, tr_y_data)
                        dirName = Merge_Path+'/'+States[i]+'/'+Doctor[j]+'/'+ Dates[h]+'/'
                        make_dir(Merge_Path, States[i], Doctor[j], Dates[h])
                        tr_x_data, tr_y_data=save(States[i],tr_x_data, tr_y_data,0, dirName)
#

for i in range(len(MOVE)):
    pre=[]
    for j in range(len(MOVE[i])):
        for a in range(len(Doctor)):
            for b in range(len(Dates)):
                dirName = CP_Path+'/'+MOVE[i][j]+'/'+ Doctor[a] + '/' + Dates[b]+'/'
                if os.path.exists(dirName):
                    f= pd.read_table(dirName + MOVE[i][j]+'.csv',sep=',',names=['Date','d_position','beconId','doctorId','rssi','txPower', 'accuracy','PHONE_beacon_id','a_x','a_y','a_z','g_x','g_y','g_z','m_x','m_y','m_z','CHECK','W1','W2','W3','W4','millis'])
                    if len(pre) !=0 :
                        pre = [pre, f]
                        pre=pd.concat(pre)
                    else :
                        pre = f
    if len(pre) != 0:
        print("1")
        tr_x_data, tr_y_data = make_matrix_move(pre, x_data, y_data, tr_x_data, tr_y_data)
        make_dir(Merge_Path,MOVE[i][j],Doctor[a], Dates[b])
        dirName = Merge_Path + '/' + MOVE[i][j] + '/' + Doctor[a] + '/' + Dates[b] + '/'
        make_dir(Merge_Path, MOVE[i][j], Doctor[a], Dates[b])
        tr_x_data, tr_y_data = save(MOVE[i][0],tr_x_data, tr_y_data,1, dirName)