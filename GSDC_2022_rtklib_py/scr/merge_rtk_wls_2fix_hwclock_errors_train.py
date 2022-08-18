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

DATA_SET = 'train'
datapath = r'D:/GSDC_Codes_Data_backup/GSDC_2022/data'

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

# Create options from trips_ph to merge rtl and wls solutions
# https://www.geeksforgeeks.org/selecting-rows-in-pandas-dataframe-based-on-conditions/
# datasets with hardware clock errors are:
options1 = list(trips_ph[0:53])  # this will be use rtk solution which is better than wls
options2 = list(trips_ph[53:54])  # this will use wls solution because the rtk solution is bad (clock errors)
options3 = list(trips_ph[54:55])  # this will be use rtk solution which is better than wls
options4 = list(trips_ph[55:56])  # this will use wls solution because the rtk solution is bad (clock errors)
options5 = list(trips_ph[56:59])  # this will be use rtk solution which is better than wls
options6 = list(trips_ph[59:60])  # this will use wls solution because the rtk solution is bad (clock errors)
options7 = list(trips_ph[60:64])  # this will be use rtk solution which is better than wls
options8 = list(trips_ph[64:66])  # this will use wls solution because the rtk solution is bad (clock errors)
options9 = list(trips_ph[66:72])  # this will be use rtk solution which is better than wls
options10 = list(trips_ph[72:73])  # this will use wls solution because the rtk solution is bad (clock errors)
options11 = list(trips_ph[73:76])  # this will be use rtk solution which is better than wls
options12 = list(trips_ph[76:77])  # this will use wls solution because the rtk solution is bad (clock errors)
options13 = list(trips_ph[77:80])  # this will be use rtk solution which is better than wls
options14 = list(trips_ph[80:81])  # this will use wls solution because the rtk solution is bad (clock errors)
options15 = list(trips_ph[81:83])  # this will be use rtk solution which is better than wls
options16 = list(trips_ph[83:84])  # this will use wls solution because the rtk solution is bad (clock errors)
options17 = list(trips_ph[84:85])  # this will be use rtk solution which is better than wls
options18 = list(trips_ph[85:86])  # this will use wls solution because the rtk solution is bad (clock errors)
options19 = list(trips_ph[86:88])  # this will be use rtk solution which is better than wls
options20 = list(trips_ph[88:89])  # this will use wls solution because the rtk solution is bad (clock errors)
options21 = list(trips_ph[89:100])  # this will be use rtk solution which is better than wls
options22 = list(trips_ph[100:101])  # this will use wls solution because the rtk solution is bad (clock errors)
options23 = list(trips_ph[101:104])  # this will be use rtk solution which is better than wls
options24 = list(trips_ph[104:105])  # this will use wls solution because the rtk solution is bad (clock errors)
options25 = list(trips_ph[105:170])  # this will be use rtk solution which is better than wls

print('\noptions:\n', options1, '\n', options2, '\n', options3, '\n', options4,  '\n', options5,
      '\n', options6, '\n', options7, '\n', options8, '\n', options9, '\n', options10,
      '\n', options11, '\n', options12, '\n', options13, '\n', options14, '\n', options15,
      '\n', options16, '\n', options17, '\n', options18, '\n', options19, '\n', options20,
      '\n', options21, '\n', options22, '\n', options23, '\n', options24, '\n', options25)

# First DataFrame for rtk solution
# df1 = pd.DataFrame({'id': ['A01', 'A02', 'A03', 'A04'],
#                     'Name': ['ABC', 'PQR', 'DEF', 'GHI']})
rtk_sub = pd.read_csv(
    r"D:\GSDC_Codes_Data_backup\GSDC_2022\data"
    r"\samsung_baseline_locations_train_rtklib_combine_noreset_el20_rnx2rtkp_train_samsung_06_20.csv")
print('\nrtk solution:\n', rtk_sub)

# Second DataFrame wls solution
# df2 = pd.DataFrame({'id': ['B05', 'B06', 'B07', 'B08'],
#                     'Name': ['XYZ', 'TUV', 'MNO', 'JKL']})
wls_sub = pd.read_csv(r"D:\GSDC_Codes_Data_backup\GSDC_2022\data\best_solutions\wls_results\train_data.csv")
print('\nwls solution:\n', wls_sub)

# Third DataFrame to get correct timestamp column
sample_sub = pd.read_csv(baseline_file)
print('\nsample solution to get correct timestamp:\n', sample_sub)

# selecting rows based on condition
rslt_df1 = rtk_sub[rtk_sub['tripId'].isin(options1)]
rslt_df2 = wls_sub[wls_sub['tripId'].isin(options2)]
rslt_df3 = rtk_sub[rtk_sub['tripId'].isin(options3)]
rslt_df4 = wls_sub[wls_sub['tripId'].isin(options4)]
rslt_df5 = rtk_sub[rtk_sub['tripId'].isin(options5)]
rslt_df6 = wls_sub[wls_sub['tripId'].isin(options6)]
rslt_df7 = rtk_sub[rtk_sub['tripId'].isin(options7)]
rslt_df8 = wls_sub[wls_sub['tripId'].isin(options8)]
rslt_df9 = rtk_sub[rtk_sub['tripId'].isin(options9)]
rslt_df10 = rtk_sub[rtk_sub['tripId'].isin(options10)]
rslt_df11 = wls_sub[wls_sub['tripId'].isin(options11)]
rslt_df12 = rtk_sub[rtk_sub['tripId'].isin(options12)]
rslt_df13 = wls_sub[wls_sub['tripId'].isin(options13)]
rslt_df14 = rtk_sub[rtk_sub['tripId'].isin(options14)]
rslt_df15 = wls_sub[wls_sub['tripId'].isin(options15)]
rslt_df16 = rtk_sub[rtk_sub['tripId'].isin(options16)]
rslt_df17 = wls_sub[wls_sub['tripId'].isin(options17)]
rslt_df18 = rtk_sub[rtk_sub['tripId'].isin(options18)]
rslt_df19 = rtk_sub[rtk_sub['tripId'].isin(options19)]
rslt_df20 = wls_sub[wls_sub['tripId'].isin(options20)]
rslt_df21 = rtk_sub[rtk_sub['tripId'].isin(options21)]
rslt_df22 = wls_sub[wls_sub['tripId'].isin(options22)]
rslt_df23 = rtk_sub[rtk_sub['tripId'].isin(options23)]
rslt_df24 = wls_sub[wls_sub['tripId'].isin(options24)]
rslt_df25 = rtk_sub[rtk_sub['tripId'].isin(options25)]


# perform the merge of rtk and wls
frames = [rslt_df1, rslt_df2, rslt_df3, rslt_df4, rslt_df5, rslt_df6, rslt_df7, rslt_df8, rslt_df9, rslt_df10,
          rslt_df11, rslt_df12, rslt_df13, rslt_df14, rslt_df15, rslt_df16, rslt_df17, rslt_df18, rslt_df19,
          rslt_df20, rslt_df21, rslt_df22, rslt_df23, rslt_df24, rslt_df25]
result = pd.concat(frames)
print('\nMerge of rtk and wls:\n', result)

# join correct time stamp column with rtk_wls lat/long column
# the default behaviour is join='outer'
# but use inner join
# final_sub = pd.concat([sample_sub.iloc[:, 0:2], result.iloc[:, 2:4]], axis=1, join='inner')
final_sub = result


# send to output file:
final_sub.to_csv('rtk_wls2fix_hw_clock_error_rnx2rtkp_train_data_samsung.csv', index=False)
print(final_sub.columns)
print()
print('\nFinal solution for submission to kaggle:\n', final_sub)


# Plot figures:
gt_truth = pd.read_csv(r"D:\GSDC_Codes_Data_backup\GSDC_2022\data\ground_truths_train.csv")
plt.figure()
plt.plot(final_sub.LongitudeDegrees, final_sub.LatitudeDegrees, label='rtk_wls2fix_train')
plt.plot(gt_truth.LongitudeDegrees, gt_truth.LatitudeDegrees, '-', label='ground_truths_train')
plt.legend()
plt.show()
