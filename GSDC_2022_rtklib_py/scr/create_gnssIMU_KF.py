"""
RTKLIB-Py:create_gnssIMU_KF.py - create csv file from all device_gnss_imu_reducedAcc

Copyright (c) 2022 Tim Everett (from rtklib_py)
Copyright (c) 2022 Akpo Siemuri
Date - 2022-06-10
"""

import os
from os.path import join, isfile

import pandas as pd

DATA_SET = 'test'
datapath = r'/home/akpo/GSDC_2022_rtklib_py/data' + '/' + DATA_SET
GPS_TO_UTC = 315964782  # second

# open output file
os.chdir(datapath)
fout = open('/home/akpo/GSDC_2022_rtklib_py/solutions/gnss_imu_solution/rtk_wls_gnssIMU_solution.csv', 'w')
fout.write('tripId,UnixTimeMillis,LongitudeDegrees,LatitudeDegrees\n')

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

        csv_file = join(folder, 'test_device_gnss_imu_reducedAcc_new_nocys_thres3.csv')
        # test_device_gnss_imu_reducedAcc_new_nocys_thres3:
        # test_device_gnss_imu_reducedAcc_new_11cys_mergePPK: same as below
        # test_device_gnss_imu_reducedAcc_new_11cys: gave the best solution on kaggle.com
        if not isfile(csv_file):
            continue

        # parse ground truth file
        with open(csv_file) as f:
            lines = f.readlines()[1:]
        flag = 0
        for line in lines:
            d = line.split(',')
            t = float(d[1])  # get time stamp
            if flag == 0:
                print('%20s,%16s' % (dataset, phone))
                flag = 1
            # write results to combined file
            fout.write('%s/%s,%.0f,%s,%s' % (dataset, phone, t, d[6], d[7]))

fout.close()

df = pd.read_csv(r"/home/akpo/GSDC_2022_rtklib_py/solutions/gnss_imu_solution/rtk_wls_gnssIMU_solution.csv")
# rtk_wls2fix_hw_clock_errors_gnssKF_test_new_nocys_thres3
# rtk_wls2fix_hw_clock_errors_gnssKF_test_new_11cys_mergePPK
print(df)
cols = ['tripId', 'UnixTimeMillis', 'LatitudeDegrees', 'LongitudeDegrees']
df = df[cols]

rtk_wls_gnssIMUKF = r"/home/akpo/GSDC_2022_rtklib_py/solutions/gnss_imu_solution/rtk_wls_gnssIMU_solution.csv"
df.to_csv(rtk_wls_gnssIMUKF, index=False)
df1 = pd.read_csv(rtk_wls_gnssIMUKF)
print(df1)
