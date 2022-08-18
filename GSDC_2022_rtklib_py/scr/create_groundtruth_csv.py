"""
RTKLIB-Py:create_groundtruth_csv.py - create csv file from all training set ground truth files

Copyright (c) 2022 Tim Everett (from rtklib_py)
Copyright (c) 2022 Akpo Siemuri
Date - 2022-05-23
"""

import os
from os.path import join, isfile

DATA_SET = 'train'
datapath = '/home/akpo/GSDC_2022_rtklib_py/data' + '/' + DATA_SET
GPS_TO_UTC = 315964782  # second

# open output file
os.chdir(datapath)
fout = open('../2few_ground_truths_train.csv', 'w')
fout.write('tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees\n')

# get list of data sets in data path
datasets = sorted(os.listdir(datapath))

# loop through data set folders
for dataset in datasets:
    if isfile(dataset):
        continue
    phones = os.listdir(join(datapath, dataset))
    for phone in phones:
        folder = join(datapath, dataset, phone)
        if isfile(folder):
            continue

        csv_file = join(folder, 'ground_truth.csv')
        if not isfile(csv_file):
            continue

        # parse ground truth file
        with open(csv_file) as f:
            lines = f.readlines()[1:]
        flag = 0
        for line in lines:
            d = line.split(',')
            t = float(d[8])  # get time stamp
            if flag == 0:
                print('%20s,%16s' % (dataset, phone))
                flag = 1
            # write results to combined file
            fout.write('%s/%s,%.0f,%s,%s\n' % (dataset, phone, t, d[2], d[3]))

fout.close()