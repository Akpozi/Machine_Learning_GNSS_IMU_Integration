import os
import time
from datetime import datetime
from os.path import join, isfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymap3d.vincenty as pmv
import pyproj as proj
from filterpy.common import Q_discrete_white_noise
from filterpy.common import Saver
from filterpy.kalman import KalmanFilter
import warnings


# Compute distance by Vincenty's formulae
def vincenty_distance(llh1, llh2):
    """
    Args:
        llh1 : [latitude,longitude] (deg)
        llh2 : [latitude,longitude] (deg)
    Returns:
        d : distance between llh1 and llh2 (m)
    """
    d, az = np.array(pmv.vdist(llh1[:, 0], llh1[:, 1], llh2[:, 0], llh2[:, 1]))

    return d


# Compute score
def calc_score(llh, llh_gt):
    """
    Args:
        llh : [latitude,longitude] (deg)
        llh_gt : [latitude,longitude] (deg)
    Returns:
        score : (m)
    """
    d = vincenty_distance(llh, llh_gt)
    score = np.mean([np.quantile(d, 0.50), np.quantile(d, 0.95)])

    return score


def csvpath_train(date, phone, file):
    """
    # Function to read csv from date//phone directory
    # date = the 1st folder titled with date
    # phone = the 2nd foler titled with phone brand
    # file = the filename of CSV data
    """
    global absolute_path
    path = os.path.join(absolute_path, 'train', date, phone, file)
    # df = pd.read_csv(path)
    return path


def csvpath_test(date, phone, file):
    """
    # Function to read csv from date//phone directory
    # date = the 1st folder titled with date
    # phone = the 2nd foler titled with phone brand
    # file = the filename of CSV data
    """
    global absolute_path
    path = os.path.join(absolute_path, 'test', date, phone, file)
    return path


train_test = 'test'
absolute_path = os.path.abspath(os.path.dirname('/home/akpo/GSDC_2022_rtklib_py/data/' + '/' + train_test))

warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')


def extract_imu_data():
    # Use the .csv file generated from the "merge_rtk_wls_2fix_hwclock_errors_test.py"
    df_baseline = pd.read_csv("/home/akpo/GSDC_2022_rtklib_py/solutions/merge_ppk_wls_solution/rtk_wls_solution.csv")
    # cys_rtk_wls2fix_hw_clock_error_test11: used to get GNSS/IMU - best on kaggle.com
    df_baseline[['date', 'phone']] = df_baseline.tripId.str.split("/", expand=True)
    df_baseline.rename(columns={'LatitudeDegrees': 'lat',
                                'LongitudeDegrees': 'long'}, inplace=True)

    # df_gt = pd.read_csv("ground_truths_train.csv")  # ground_truths_train
    # df_gt[['date', 'phone']] = df_gt.tripId.str.split("/", expand=True)
    # df_gt.rename(columns={'LatitudeDegrees': 'lat',
    #                       'LongitudeDegrees': 'long'}, inplace=True)

    # ########## Input parameters ###############################

    DATA_SET = train_test
    datapath = r'/home/akpo/GSDC_2022_rtklib_py/data/'

    ############################################################
    # get list of data sets in data path
    os.chdir(join(datapath, DATA_SET))
    trips = sorted(os.listdir())
    # Unique sorted list of experiment dates folders
    # date_foldrs = sorted(df_baseline['date'].unique())
    # print(date_foldrs)

    # setup projections conversion long/lat from-to x/y
    wgs = proj.Proj(init='epsg:4326')  # assuming you're using WGS84 geographic
    bng = proj.Proj(init='epsg:27700')  # use a locally appropriate projected CRS (British national)
    ecef = proj.Proj(init='epsg:4978')  # This 4978 is ECEF

    # Read datasets of corresponding date/phones
    file = 'device_imu.csv'
    newfile = 'test_device_gnss_imu_reducedAcc_new_nocys_thres3.csv'

    # Measurement matrix (Jacobian in EKF)
    trips_ph = []
    for trip in trips:
        phones = os.listdir(trip)
        # loop through phone folders
        for phone in phones:
            # check for valid folder and file
            dt = 0.001 if phone == 'XiaomiMi8' else 0.001
            folder = join(trip, phone)
            if isfile(folder):
                continue
            trip_phone = trip + '/' + phone
            # print(trip_phone)
            print('\nProcessing:', trip_phone)

            # Select the date/phone data only
            # phone = df_baseline['phone'][df_baseline['date'] == date].any()
            dfs = df_baseline.loc[(df_baseline['date'] == trip) & (df_baseline['phone'] == phone)]  # rtk_wls
            dfs.reset_index(inplace=True)

            # For ground truth
            # dfs_gt = df_gt.loc[(df_gt['date'] == trip) & (df_gt['phone'] == phone)]  # ground truth
            # dfs_gt.reset_index(inplace=True)

            xg, yg = proj.transform(wgs, bng, dfs['long'], dfs['lat'])  # GNSS data in x,y
            # xgt, ygt = proj.transform(wgs, bng, dfs_gt['long'], dfs_gt['lat'])  # GNSS ground truth data in x,y

            # Reading the corresponding IMU accelerometer data
            dfx = pd.read_csv(csvpath_test(trip, phone, file),
                              usecols=['MessageType', 'utcTimeMillis', 'MeasurementX', 'MeasurementY'])
            dfx = dfx[dfx['MessageType'] == 'UncalAccel']
            dfx.reset_index(inplace=True)
            dfx = dfx.drop(columns='index')

            # Selecting the range IMU timestamp(s) which are projected from GNSS
            ax, ay = [], []

            for w in range(1, dfs.shape[0]):
                t0 = dfs['UnixTimeMillis'][w - 1]
                t1 = dfs['UnixTimeMillis'][w]
                indx = dfx.index[(dfx['utcTimeMillis'] >= t0) & (dfx['utcTimeMillis'] <= t1)]
                ax.append(np.mean(dfx['MeasurementX'][indx]))
                ay.append(np.mean(dfx['MeasurementY'][indx]))

                if w == dfs.shape[0] - 1:
                    ax.append(dfx['MeasurementX'].iloc[-1])
                    ay.append(dfx['MeasurementY'].iloc[-1])

            # Saving the output as dataframe
            dfo = pd.DataFrame()
            dfo['unixTime'] = dfs['UnixTimeMillis']
            dfo['ax'] = ax
            dfo['ay'] = ay
            # dis = len(dfo['ay'])
            # print('dfo:', dis)

            x_dim = 6
            z_dim = 4
            # sigma_x = 5.0  # position SD m
            # sigma_v = 0.6 if phone == 'XiaomiMi8' else 0.1  # velocity SD m/s
            # Creating KF object
            dim_x = x_dim
            dim_z = z_dim
            kflt = KalmanFilter(dim_x, dim_z)
            # kflt.Q = sigma_v ** 2 * np.eye(3)  # Process noise
            kflt.Q = Q_discrete_white_noise(dim=2, dt=dt, var=9e11, block_size=3, order_by_dim=True)
            # kflt.R = sigma_x ** 2 * np.eye(dim_z)
            kflt.R = np.eye(dim_z)
            kflt.H = np.array([[1, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0, 1]])
            # kflt.P = sigma_x ** 2 * np.eye(dim_x)  # State covariance
            kflt.P = np.eye(dim_x)
            kflt.F = np.array([[1, dt, 0.5 * (dt ** 2), 0, 0, 0],
                               [0, 1, dt, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 1, dt, 0.5 * (dt ** 2)],
                               [0, 0, 0, 0, 1, dt],
                               [0, 0, 0, 0, 0, 1]])
            kflt.x[0] = xg[0]
            kflt.x[2] = dfo['ax'][0]
            kflt.x[3] = yg[0]
            kflt.x[5] = dfo['ay'][0]

            # print('Shape: ', dfo.shape)

            # Looping KF
            xkf, ykf = [], []
            saver_kf = Saver(kflt)
            for k in range(0, dfo.shape[0]):
                # Measurement z.
                z = np.array([[xg[k]], [dfo['ax'][k]], [yg[k]], [dfo['ay'][k]]])
                kflt.predict()
                kflt.update(z)
                # print('KFLT: ', kflt.x[0])

                xkf.append(kflt.x[0])
                ykf.append(kflt.x[3])
                # print('size of xkf', len(xkf))
                saver_kf.save()

            # KF-RTS smoother
            xs_kf = np.array(saver_kf.x)
            # print('Tests:', xs_kf)
            covs_kf = np.array(saver_kf.P)
            Mrts_kf, Prts_kf, _, _ = kflt.rts_smoother(xs_kf, covs_kf)
            # print('Mrts_kf:\n', Mrts_kf)
            # end_df = len(Mrts_kf)
            # print('size of Mrts_kf: ', end_df)

            # print('xkf ', xkf)
            # print('ykf ', ykf)
            # KF output in long/lat:
            dfo['kf_long'], dfo['kf_lat'] = proj.transform(bng, wgs, xkf, ykf)  # KF output in long/lat
            # print(dfo['kf_long'], dfo['kf_lat'])

            # # numpy get values in array of arrays for array of indices: Used to get desired Mrts_kf[0]/Mrts_kf[3] values
            # https://stackoverflow.com/questions/33800210/numpy-get-values-in-array-of-arrays-of-arrays-for-array-of-indices
            # zeros_x = np.zeros(end_df, dtype=np.int64)  # get Mrts_kf[0] values
            # print('Zeros:\n', zeros_x)
            # desired_array_x = np.array([Mrts_kf[x][y] for x, y in enumerate(zeros_x)])
            # print('desired_array_x:', len(desired_array_x))
            #
            # threes_y = np.zeros(end_df, dtype=np.int64)+3  # get Mrts_kf[3] values
            # print('Zeros:\n', threes_y)
            # desired_array_y = np.array([Mrts_kf[x][y] for x, y in enumerate(threes_y)])
            # print('desired_array_y:', desired_array_y)

            # print('Mrts_kf_X ', Mrts_kf[:, 0])
            # print('Mrts_kf_Y ', Mrts_kf[:, 3])
            # RTD output in long/lat:
            dfo['rts_long'], dfo['rts_lat'] = proj.transform(bng, wgs, Mrts_kf[:, 0],
                                                             Mrts_kf[:, 3])  # RTD output: long/lat
            # print(dfo['rts_long'], dfo['rts_lat'])

            # Save all IMU outputs data to file:
            dfo.to_csv(csvpath_test(trip, phone, newfile))  # Save output GNSS/IMU
            print()

            # Ground truth, rtk_wls, kf, and rts kf
            # llh_gt = dfs_gt[['lat', 'long']].to_numpy()
            llh_rtk_wls = dfs[['lat', 'long']].to_numpy()
            llh_kf = dfo[['kf_lat', 'kf_long']].to_numpy()
            llh_rts = dfo[['rts_lat', 'rts_long']].to_numpy()
            # llh_bl = np.array(pm.ecef2geodetic(x_bl[:, 0], x_bl[:, 1], x_bl[:, 2])).T

            # balanced_signal = llh_rtk_wls - np.mean(llh_rtk_wls)
            # spectrum = np.fft.rfft(balanced_signal)

            # print('length of llh_gt:', len(llh_gt))
            print('length of llh_rtk_wls:', len(llh_rtk_wls))
            print('length of llh_kf:', len(llh_kf))
            print('length of llh_rts:', len(llh_rts))
            # print()

            # Distance from ground truth
            # vd_rtk_wls = vincenty_distance(llh_rtk_wls, llh_gt)
            # vd_kf = vincenty_distance(llh_kf, llh_gt)
            # vd_rts = vincenty_distance(llh_rts, llh_gt)

            # Score
            # score_bl = calc_score(llh_rtk_wls, llh_gt)
            # score_kf = calc_score(llh_kf, llh_gt)
            # score_rts = calc_score(llh_rts, llh_gt)

            # print(f'Score rtk_wls Baseline: {score_bl:.4f} [m]')
            # print(f'Score KF:               {score_kf:.4f} [m]')
            # print(f'Score RTS KF:           {score_rts:.4f} [m]')

            # Plot distance error
            # plt.figure()
            # plt.title('Distance error')
            # plt.ylabel('Distance error [m]')
            # plt.plot(vd_rtk_wls, label=f'Score rtk_wls Baseline: {score_bl:.4f} m')
            # plt.plot(vd_kf, label=f'Score KF, Score:             {score_kf:.4f} m')
            # plt.plot(vd_rts, label=f'Score RTS KF, Score:        {score_rts:.4f} m')
            # plt.legend()
            # plt.grid()
            # plt.ylim([0, 30])

            # # Plot figures: X and T
            # plt.figure()
            # # plt.plot(dfo['unixTime'], xgt, '-', label='groundTruth')
            # plt.plot(dfo['unixTime'], xg, label='gnss')
            # plt.plot(dfo['unixTime'], xkf, '-', label='KF fusion')
            # plt.plot(dfo['unixTime'], Mrts_kf[:, 0], '-', label='RTS KF')
            # plt.legend()
            #
            # # Plot figures: Y and T
            # plt.figure()
            # # plt.plot(dfo['unixTime'], ygt, '-', label='groundTruth')
            # plt.plot(dfo['unixTime'], yg, label='gnss')
            # plt.plot(dfo['unixTime'], ykf, '-', label='KF fusion')
            # plt.plot(dfo['unixTime'], Mrts_kf[:, 3], '-', label='RTS KF')
            # plt.legend()
            #
            # # Plot figures:
            # plt.figure()
            # # plt.plot(xgt, ygt, '-', label='groundTruth')
            # plt.plot(xg, yg, label='gnss')
            # plt.plot(xkf, ykf, '-', label='KF fusion')
            # plt.plot(Mrts_kf[:, 0], Mrts_kf[:, 3], '-', label='RTS KF')
            # plt.legend()
            # plt.show()

            # # Plot figures:
            # plt.figure()
            # plt.plot(dfs_gt['long'], dfs_gt['lat'], '-', label='groundTruth')
            # plt.plot(dfs['long'], dfs['lat'], label='gnss')
            # plt.plot(dfo['kf_long'], dfo['kf_lat'], '-', label='KF fusion')
            # plt.plot(dfo['rts_long'], dfo['rts_lat'], '-', label='RTS KF')
            # plt.legend()
            # plt.show()

        # https://gis.stackexchange.com/questions/212723/how-can-i-convert-lon-lat-coordinates-to-x-y
        # https://gis.stackexchange.com/questions/386070/converting-from-epsg4978-to-epsg4326-using-pyproj-always-produces-0-0-latitude


# This part is the first to execute when script is ran. It times the execution time of the function
datetime.now().strftime('%Y-%m-%d %H:%M:%S')
start = time.time()  # StartTime
print(
    "\nScript started at " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " OR in seconds: " + str(
        start) + " seconds")
print()
extract_imu_data()  # Run the function to perform GNSS/IMU fusion
end = time.time()  # EndTime
print("Script ended at " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " OR in seconds: " + str(end) + " seconds")
print("Script completed in " + str((end - start)) + " seconds OR " + str((end - start) / 60) + " minutes")
print(' ')
print('Done! Download completed')
