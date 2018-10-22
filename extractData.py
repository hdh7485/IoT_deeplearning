import csv
import shutil
import os
import numpy
import random
from joblib import Parallel, delayed
import multiprocessing

# user configuration


BASE = '../new_data/'

MIN_ROWS = 32
#MAX_VAL = 200.0

# DATA_SIZE = 38
# DATA_SIZE = 10
NUM_LABELS = 0
NUM_CHANNELS=0
#
Dates = '20170331, 20170403, 20170404, 20170405, 20170406, 20170407, 20170410, 20170411, 20170412,20170413, 20170414, 20170417, 20170419, 20170420, 20170421, 20170424, 20170425, 20170426, 20170427, 20170428'
Doctor = '00000004,00000049'

MOVE = 'C1_C2, C1_C3, C1_C4, C1_C5, C1_C6, C1_C7, C1_C8, C1_O1, C1_O2, C1_P1, C1_P2, C1_T1, C1_Z1, C1_A1, C1_A2,' \
       'C2_C1, C2_C3, C2_C4, C2_C5, C2_C6, C2_C7, C2_C8, C2_O1, C2_O2, C2_P1, C2_P2, C2_T1, C2_Z1, C2_A1, C2_A2,' \
       'C3_C1, C3_C2, C3_C4, C3_C5, C3_C6, C3_C7, C3_C8, C3_O1, C3_O2, C3_P1, C3_P2, C3_T1, C3_Z1, C3_A1, C3_A2,' \
       'C4_C1, C4_C2, C4_C3, C4_C5, C4_C6, C4_C7, C4_C8, C4_O1, C4_O2, C4_P1, C4_P2, C4_T1, C4_Z1, C4_A1, C4_A2, ' \
       'C5_C1, C5_C2, C5_C3, C5_C4, C5_C6, C5_C7, C5_C8, C5_O1, C5_O2, C5_P1, C5_P2, C5_T1, C5_Z1, C5_A1, C5_A2, ' \
       'C6_C1, C6_C2, C6_C3, C6_C4, C6_C5, C6_C7, C6_C8, C6_O1, C6_O2, C6_P1, C6_P2, C6_T1, C6_Z1, C6_A1, C6_A2, ' \
       'C7_C1, C7_C2, C7_C3, C7_C4, C7_C5, C7_C6, C7_C8, C7_O1, C7_O2, C7_P1, C7_P2, C7_T1, C7_Z1, C7_A1, C7_A2,' \
       'C8_C1, C8_C2, C8_C3, C8_C4, C8_C5, C8_C6, C8_C7, C8_O1, C8_O2, C8_P1, C8_P2, C8_T1, C8_Z1, C8_A1, C8_A2,' \
       'O1_C1, O1_C2, O1_C3, O1_C4, O1_C5, O1_C6, O1_C7, O1_C8, O1_O2, O1_P1, O1_P2, O1_T1, O1_Z1, O1_A1, O1_A2,' \
       'O2_C1, O2_C2, O2_C3, O2_C4, O2_C5, O2_C6, O2_C7, O2_C8, O2_O1, O2_P1, O2_P2, O2_T1, O2_Z1, O2_A1, O2_A2,' \
       'P1_C1, P1_C2, P1_C3, P1_C4, P1_C5, P1_C6, P1_C7, P1_C8, P1_O1, P1_O2, P1_P2, P1_T1, P1_Z1, P1_A1, P1_A2,' \
       'P2_C1, P2_C2, P2_C3, P2_C4, P2_C5, P2_C6, P2_C7, P2_C8, P2_O1, P2_O2, P2_P1, P2_T1, P2_Z1, P2_A1, P2_A2,' \
       'T1_C1, T1_C2, T1_C3, T1_C4, T1_C5, T1_C6, T1_C7, T1_C8, T1_O1, T1_O2, T1_P1, T1_P2, T1_Z1, T1_A1, T1_A2,' \
       'Z1_C1, Z1_C2, Z1_C3, Z1_C4, Z1_C5, Z1_C6, Z1_C7, Z1_C8, Z1_O1, Z1_O2, Z1_P1, Z1_P2, Z1_T1, Z1_A1, Z1_A2,' \
       'A1_C1, A1_C2, A1_C3, A1_C4, A1_C5, A1_C6, A1_C7, A1_C8, A1_O1, A1_O2, A1_P1, A1_P2, A1_T1, A1_Z1, A1_A2,' \
       'A2_C1, A2_C2, A2_C3, A2_C4, A2_C5, A2_C6, A2_C7, A2_C8, A2_O1, A2_O2, A2_P1, A2_P2, A2_T1, A2_Z1, A2_A1'

States = 'C1,C2,C3,C4,C5,C6,C7,C8,O1,O2,P1,P2,T1,mC1,mC2,mC3,mC4,mC5,mC6,mC7,mC8,mO1,mO2,mP1,mP2,mT1'
Labels = {}


def get_files(states=None, moves=None, dates = None, doctors=None):
    if states is None or len(states) == 0:
        states = States

    if dates is None or len(dates) == 0:
        dates = Dates

    if moves is None or len(moves) == 0:
        moves = MOVE

    if doctors is None or len(doctors) == 0:
        doctors = Doctor

    filenames = []
    filelabels = []

    dirName = BASE

    for state in states:
        dirName1 = os.path.join(dirName, state)
        if os.path.exists(dirName1):
            for doctor in doctors:
                dirName2 = os.path.join(dirName1, doctor)
                if os.path.exists(dirName2):
                    for date in dates:
                        dirName3 = os.path.join(dirName2, date)
                        if os.path.exists(dirName3):
                            for fn in os.listdir(dirName3):
                                filenames.append(dirName1 +'/' + fn)
                                filelabels.append(Labels[state])

    for move in moves:
        dirName1 = os.path.join(dirName, move)
        if os.path.exists(dirName1):
            for doctor in doctors:
                dirName2 = os.path.join(dirName1, doctor)
                if os.path.exists(dirName2):
                    for date in dates:
                        dirName3 = os.path.join(dirName2, date)
                        if os.path.exists(dirName3):
                            for fn in os.listdir(dirName3):
                                filenames.append(dirName1 + '/' + fn)
                                filelabels.append(Labels['m'+move.split('_')[0]])

    print "found %d files" % len(filenames)
    return filenames, filelabels




def fileReader(fn):
    reader = csv.reader(open(fn, "rb"), delimiter=',')
    x = list(reader)
    result = numpy.array(x).astype('float')
    print '%s has %d rows.' % (fn, result.shape[0])
    return result

def extract_data(states=None, labels=None, moves=None, doctors=None, dates=None, numRows=None, numData=None, mode=None, ONED=False, DATA_SIZE=None):

    NUM_LABELS = len(set(labels))
    print NUM_LABELS
    for s, l in zip(states, labels):
        Labels[s] = l

    if numRows is None:
        numRows = 30 * 10 # frames/sec * sec

    if numData is None:
        numData = 1

    filenames, filelabels  = get_files(states,moves,dates,doctors)

    results = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(fileReader)(fn) for fn in filenames)

    if ONED:
        data = numpy.ndarray(shape=(numData, numRows, 1, DATA_SIZE), dtype=numpy.float32)
    else:
        data = numpy.ndarray(shape=(numData, numRows, DATA_SIZE, 1), dtype=numpy.float32)
    dataLabel = numpy.zeros(shape=(numData, NUM_LABELS), dtype=numpy.float32)

    numDataCount = 0
    while numDataCount < numData:
        res_idx = random.randint(0, len(results) - 1)
        max_row_idx = results[res_idx].shape[0] - numRows

        if max_row_idx < MIN_ROWS:  # this value should be at least 0
            continue

        if mode is 'train':
            row_idx = random.randint(0, max_row_idx / 2)
            # print 'train block %d ~ %d' % (0, max_row_idx / 2)
        elif mode is 'validate':
            row_idx = random.randint(max_row_idx / 2, max_row_idx * 3 / 4)
            # print 'validate block %d ~ %d' % (max_row_idx / 2, max_row_idx * 3 / 4)
        elif mode is 'test':
            row_idx = random.randint(max_row_idx * 3 / 4, max_row_idx)
            # print 'test block %d ~ %d' % (max_row_idx * 3 / 4, max_row_idx)
        else:
            row_idx = random.randint(0, max_row_idx)

        if ONED:
            data[numDataCount, :, 0, :]= results[res_idx][row_idx:row_idx + numRows, 0:NUM_CHANNELS]
        else:
            # data[numDataCount, :, :, 0] = results[res_idx][row_idx:row_idx + numRows, :]
            data[numDataCount, :, :, 0] = results[res_idx][row_idx:row_idx + numRows, 0:DATA_SIZE]

        filelabels[res_idx]=int(filelabels[res_idx])

        dataLabel[numDataCount, filelabels[res_idx]] = 1.0
        # dataLabel[numDataCount, labels[res_idx]] = 1.0
        numDataCount = numDataCount + 1
    return data, dataLabel

def extract_data_oned(states=None, labels=None, moves=None, doctors=None, dates=None, numRows=None, numData=None, mode=None, NUM_CHANNELS=None,
                     ONED=True, DATA_SIZE=None):

    NUM_LABELS = len(set(labels))
    print NUM_LABELS
    for s, l in zip(states, labels):
        Labels[s] = l


    if numRows is None:
        numRows = 30 * 10  # frames/sec * sec

    if numData is None:
        numData = 1

    filenames, filelabels = get_files(states,moves,dates,doctors)

    results = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(fileReader)(fn) for fn in filenames)

    if ONED:
        data = numpy.ndarray(shape=(numData, numRows, 1, NUM_CHANNELS), dtype=numpy.float32)
    else:
        data = numpy.ndarray(shape=(numData, numRows, DATA_SIZE, 1), dtype=numpy.float32)
    dataLabel = numpy.zeros(shape=(numData, NUM_LABELS), dtype=numpy.float32)

    numDataCount = 0
    while numDataCount < numData:
        res_idx = random.randint(0, len(results) - 1)
        max_row_idx = results[res_idx].shape[0] - numRows

        if max_row_idx < MIN_ROWS:  # this value should be at least 0
            continue

        if mode is 'train':
            row_idx = random.randint(0, max_row_idx / 2)
            # print 'train block %d ~ %d' % (0, max_row_idx / 2)
        elif mode is 'validate':
            row_idx = random.randint(max_row_idx / 2, max_row_idx * 3 / 4)
            # print 'validate block %d ~ %d' % (max_row_idx / 2, max_row_idx * 3 / 4)
        elif mode is 'test':
            row_idx = random.randint(max_row_idx * 3 / 4, max_row_idx)
            # print 'test block %d ~ %d' % (max_row_idx * 3 / 4, max_row_idx)
        else:
            row_idx = random.randint(0, max_row_idx)

        if ONED:
            data[numDataCount, :, 0, :] = results[res_idx][row_idx:row_idx + numRows, 0:NUM_CHANNELS]
        else:
            # data[numDataCount, :, :, 0] = results[res_idx][row_idx:row_idx + numRows, :]
            data[numDataCount, :, :, 0] = results[res_idx][row_idx:row_idx + numRows, 0:DATA_SIZE]

        filelabels[res_idx] = int(filelabels[res_idx])

        dataLabel[numDataCount, filelabels[res_idx]] = 1.0
        # dataLabel[numDataCount, labels[res_idx]] = 1.0
        numDataCount = numDataCount + 1


    # maxVal = data.max(axis=3).max(axis=2).max(axis=1).max(axis=0)
    # minVal = data.min(axis=3).min(axis=2).min(axis=1).min(axis=0)
    # print "maxVal = %f, minVal = %f" % (maxVal, minVal)
    # data = data / MAX_VAL
    return data, dataLabel


    if __name__ == '__main__':
        data, dataLabel = extract_data(numRows = 300, numData = 1000)

        dataLabel[numDataCount, filelabels[res_idx]] = 1.0

        numDataCount = numDataCount + 1
    return data, dataLabel
