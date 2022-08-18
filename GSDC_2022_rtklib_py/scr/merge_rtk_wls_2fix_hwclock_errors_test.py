"""
count_clock_errors.py - count hardware clock discontinuities in raw logs
merge rtk solution with wls (for datasets with hardware clock discontinuities)
Akpo Siemuri: 2022-06-09
"""

import os
from os.path import join, isfile
import pandas as pd
import matplotlib.pyplot as plt

# ########## Input parameters ###############################

DATA_SET = 'test'
datapath = r'/home/akpo/GSDC_2022_rtklib_py/data/'

############################################################
# get list of data sets in data path
os.chdir(join(datapath, DATA_SET))
trips = os.listdir()

if DATA_SET == 'train':
    baseline_file = 'ground_truths_train.csv'
else:  # 'test'
    baseline_file = 'sample_submission.csv'

# loop through data set folders
trips_ph = []
for trip in trips:
    if isfile(trip):
        continue
    phones = os.listdir(trip)
    # loop through phone folders
    for phone in phones:
        # check for valid folder and file
        folder = join(trip, phone)
        if isfile(folder):
            continue
        trip_phone = trip + '/' + phone
        # print(trip_phone)
        trips_ph.append(trip_phone)
print('\nfolders_phones: ', trips_ph)
print('\nfolders_phones: ', len(trips_ph))

# move to train/test directory:
os.chdir(datapath)

# First DataFrame for rtk solution
rtk_sub = pd.read_csv(
    r"/home/akpo/GSDC_2022_rtklib_py/solutions/ppk_solution"
    r"/test_py0510_combined_noreset_MLPathPredict_08_18.csv")
# sorted_thres3_baseline_locations_test_py0510_combined_noreset_cnr20_el10_nocycle_thres3_06_22: 20220622
# rtk_wls2fix_hw_clock_errors_gnssKF_test_new_11cys_mergePPK: 20220621
# sorted_cys_baseline_locations_test__py0510_combined_noreset_cnr20_el10_samsung_cycle_06_21: 20220621
print('\nrtk solution:\n', rtk_sub)

# Second DataFrame wls solution
wls_sub = pd.read_csv(r"/home/akpo/GSDC_2022_rtklib_py/solutions/wls_solution/submission.csv")
print('\nwls solution:\n', wls_sub)

# Third DataFrame to get correct timestamp column
sample_sub = pd.read_csv(baseline_file)
print('\nsample solution to get correct timestamp:\n', sample_sub)

# Create options from trips_ph to merge rtl and wls solutions
# https://www.geeksforgeeks.org/selecting-rows-in-pandas-dataframe-based-on-conditions/
# datasets with hardware clock errors are:
# options2 = ['2021-08-12-US-MTV-1/GooglePixel4'] and
# options4 = ['2022-04-25-US-OAK-2/GooglePixel4']
options1 = list(trips_ph[0:2])  # this will be use rtk solution which is better than wls
options2 = list(trips_ph[2:3])  # this will use wls solution because the rtk solution is bad (clock errors)
options3 = list(trips_ph[3:35])  # this will be use rtk solution which is better than wls
options4 = list(trips_ph[35:36])  # this will use wls solution because the rtk solution is bad (clock errors)
print('\noptions:\n', options1, '\n', options2, '\n', options3, '\n', options4)

# selecting rows based on condition
rslt_df1 = rtk_sub[rtk_sub['tripId'].isin(options1)]
rslt_df2 = wls_sub[wls_sub['tripId'].isin(options2)]
rslt_df3 = rtk_sub[rtk_sub['tripId'].isin(options3)]
rslt_df4 = wls_sub[wls_sub['tripId'].isin(options4)]

# perform the merge of rtk and wls
frames = [rslt_df1, rslt_df2, rslt_df3, rslt_df4]
result = pd.concat(frames)
print('\nMerge of rtk and wls:\n', result)

# join correct time stamp column with rtk_wls lat/long column
# the default behaviour is join='outer'
# but use inner join
final_sub = pd.concat([sample_sub.iloc[:, 0:2], result.iloc[:, 2:4]], axis=1, join='inner')

# send to output file:
final_sub.to_csv('/home/akpo/GSDC_2022_rtklib_py/solutions/merge_ppk_wls_solution/rtk_wls_solution.csv', index=False)
print(final_sub.columns)
print()
print('\nFinal solution for submission to kaggle:\n', final_sub)
