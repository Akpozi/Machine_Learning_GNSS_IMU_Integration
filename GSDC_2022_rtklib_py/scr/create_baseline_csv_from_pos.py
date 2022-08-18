"""
RTKLIB-Py: create_baseline_csv_from_pos.py -  Create csv file PPK solution files using timestamps in reference file
data set with RTKLIB and/or rtklib-py.

Copyright (c) 2022 Tim Everett (from rtklib_py)
Copyright (c) 2022 Akpo Siemuri
Date - 2022-05-23
"""

import os
from os.path import join, isfile
import numpy as np
from datetime import date

# ########## Input parameters ###############################

DATA_SET = 'test'
SOL_TAG = '_py0510_combined_noreset_MLPathPredict'
datapath = '/home/akpo/GSDC_2022_rtklib_py/data'
hdrlen = 1  # 25 for RTKLIB, 1 for rtklib-py

# Also make sure the appropriate reference file is in the datapath
#  test: sample_submission.csv - provided in Google data
# train: ground_truths_train.csv - created with crete_ground_truths.py

############################################################

GPS_TO_UTC = 315964782  # second

# get timestamps from existing baseline file
os.chdir(datapath)
if DATA_SET == 'train':
    baseline_file = 'ground_truths_train.csv'
else:  # 'test'
    baseline_file = 'sample_submission.csv'
base_txt = np.genfromtxt(baseline_file, delimiter=',', invalid_raise=False,
                         skip_header=1, dtype=str)
msecs_base = base_txt[:, 1].astype(np.int64)
phones_base = base_txt[:, 0]

# open output file
fout = open('/home/akpo/GSDC_2022_rtklib_py/solutions/ppk_solutions/' + DATA_SET + SOL_TAG + '_' + date.today().strftime("%m_%d") + '.csv', 'w')
fout.write('tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees\n')

# get list of data sets in data path
os.chdir(join(datapath, DATA_SET))
trips = sorted(os.listdir())

# loop through data set folders
ix_b = 0
for trip in trips:
    if isfile(trip):
        continue
    phones = os.listdir(trip)  # ['GooglePixel4', 'GooglePixel4XL', 'SamsungGalaxyS20Ultra', 'XiaomiMi8']
    # loop through phone folders
    for phone in phones:
        # check for valid folder and file
        folder = join(trip, phone)
        if isfile(folder):
            continue
        trip_phone = trip + '/' + phone
        print(trip_phone)

        ix_b = np.where(phones_base == trip_phone)[0]
        sol_path = join(folder, 'supplemental', 'gnss_log' + SOL_TAG + '.pos')
        if isfile(sol_path):
            # parse solution file
            fields = np.genfromtxt(sol_path, invalid_raise=False, skip_header=hdrlen)
            if int(fields[0, 1]) > int(fields[-1, 1]):  # invert if backwards solution
                fields = fields[::-1]
            pos = fields[:, 2:5]
            # qs = fields[:,5].astype(int)
            # nss = fields[:,6].astype(int)
            # acc = fields[:,7:13]
            msecs = (1000 * (fields[:, 0] * 7 * 24 * 3600 + fields[:, 1])).astype(np.int64)
            msecs += GPS_TO_UTC * 1000
        else:
            print('File not found: ', sol_path)
            msecs = msecs_base.copy()
            pos = acc = np.zeros((len(msecs), 3))
            qs = nss = np.zeros(len(msecs))

        # interpolate to baseline timestamps to fill in missing samples
        llhs = []
        stds = []
        for j in range(6):
            if j < 3:
                llhs.append(np.interp(msecs_base[ix_b], msecs, pos[:, j]))
        #     stds.append(np.interp(msecs_b, msecs, acc[:,j],
        #                         left=1000, right=1000))
        # qsi = np.interp(msecs_b, msecs, qs)
        # nssi = np.interp(msecs_b, msecs, nss)

        # # write results to combined file
        for i in range(len(ix_b)):
            fout.write('%s,%d,%s,%s\n' %
                       (trip_phone, msecs_base[ix_b[i]], llhs[0][i], llhs[1][i]))

fout.close()
