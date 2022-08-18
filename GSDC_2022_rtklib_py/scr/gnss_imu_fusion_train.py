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


train_test = 'train'
absolute_path = os.path.abspath(os.path.dirname('D:/GSDC_Codes_Data_backup/GSDC_2022/data' + '/' + train_test))

warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')


def extract_imu_data():
    df_baseline = pd.read_csv(
        "rtk_wls2fix_hw_clock_error_rnx2rtkp_train_data.csv")
    # train_amode3thres3_baseline_locations_train_py0510_combined_noreset_cnr20_el10_armode3_thres3_cnr_min20train_07_07
    # rtk_wls2fix_hw_clock_error_rnx2rtkp_train_data
    # train_thres3_baseline_locations_train_py0510_combined_noreset_cnr20_el10_armode0_thres3_cnr_min40train_06_27
    # sorted_train_thres3_baseline_locations_train_py0510_combined_noreset_cnr20_el10_nocycle_thres3_train_06_22
    # sorted_train_thres3_baseline_locations_train_py0510_combined_noreset_cnr20_el20_nocycle_thres3_train_06_27
    # new_train_submission_: same as new_train_submission: SavgolFilter + outlier: 1st best so far
    # train_submission_KNN
    # submission_best12: smart_ensemble solution of 1st & 2nd best
    # train_submission_kf: KF only:
    # train_submission: SavgolFilter + outlier: 1st best so far
    # new_train_submission: SavgolFilter only: 2nd best so far
    # train_final_submission: outlier only: poor
    # rtk_wls2fix_hw_clock_error_rnx2rtkp_train_data: 3rd best so far, better than anything below - same as _06_17
    # sorted_Nbaseline_locations_train_rtklib_combine_noreset_el20_vel_rnx2rtkp_train_06_17: same as _06_16
    # sorted_NNbaseline_locations_train_rtklib_combine_noreset_el20_rnx2rtkp_train_06_16 : v best so far
    # rtk_wls2fix_hw_clock_error_use_train_data : best so far
    # train_comnoreset_el20_rnx2rtkp_smoothing_06_17
    # sorted_baseline_locations_train__py0510_combined_noreset_cnr20_el20_thresdop5_06_17
    # train_rtk_wls2fix_hw_clck_errors_rnx2rtkp_bais
    # train_rtk_wls2fix_hw_clck_errors_bais
    # rtk_wls2fix_hw_clock_error_new_train_data
    # sorted_baseline_locations_train_rtklib_combine_noreset_snr52_l1l5_armode3_train_06_15

    # df_wls = pd.read_csv("train_BL_output.csv")
    #
    # df_wls[['date', 'phone']] = df_wls.tripId.str.split("/", expand=True)
    # df_wls.rename(columns={'LatitudeDegrees': 'lat',
    #                        'LongitudeDegrees': 'long'}, inplace=True)

    df_baseline[['date', 'phone']] = df_baseline.tripId.str.split("/", expand=True)
    df_baseline.rename(columns={'LatitudeDegrees': 'lat',
                                'LongitudeDegrees': 'long'}, inplace=True)

    df_gt = pd.read_csv("ground_truths_train.csv")  # ground_truths_train
    df_gt[['date', 'phone']] = df_gt.tripId.str.split("/", expand=True)
    df_gt.rename(columns={'LatitudeDegrees': 'lat',
                          'LongitudeDegrees': 'long'}, inplace=True)
    # xx_gt = df_gt[['LongitudeDegrees', 'LatitudeDegrees']].to_numpy()

    # ########## Input parameters ###############################

    DATA_SET = train_test
    datapath = r'D:/GSDC_Codes_Data_backup/GSDC_2022/data'

    ############################################################
    # get list of data sets in data path
    os.chdir(join(datapath, DATA_SET))
    trips = sorted(os.listdir())
    # print(trips)
    # Unique sorted list of experiment dates folders
    # date_foldrs = sorted(df_baseline['date'].unique())
    # print(date_foldrs)

    # setup projections conversion long/lat from-to x/y
    wgs = proj.Proj(init='epsg:4326')  # assuming you're using WGS84 geographic
    bng = proj.Proj(init='epsg:27700')  # use a locally appropriate projected CRS (British national)
    ecef = proj.Proj(init='epsg:4978')  # This 4978 is ECEF

    # Read datasets of corresponding date/phones
    file = 'device_imu.csv'
    newfile = 'train_device_gnss_imu_reducedAcc_new.csv'
    dt = 0.001
    # States vector is x = [x vx ax y vy ay]
    # dt = 0.001
    # px_symb = sympy.Symbol('p_x')
    # py_symb = sympy.Symbol('p_y')
    # vx_symb = sympy.Symbol('v_x')
    # vy_symb = sympy.Symbol('v_y')
    # ax_symb = sympy.Symbol('a_x')
    # ay_symb = sympy.Symbol('a_y')
    # omega_x = sympy.Symbol('omega_x')
    # omega_y = sympy.Symbol('omega_y')
    # dt_symb = sympy.Symbol('dt')

    # States vector is x = [x vx ax y vy ay]
    # vectx = Matrix([[px_symb], [vx_symb], [ax_symb],
    #                 [py_symb], [vy_symb], [ay_symb]])

    # define the state vector x_k
    # x, y = dynamicsymbols('x, y')
    # t = sympy.Symbol('t')
    # x_dot = x.diff(t)
    # x_ddot = x_dot.diff(t)
    # y_dot = y.diff(t)
    # y_ddot = y_dot.diff(t)

    # Measurements
    # gx = Matrix([[px_symb], [ax_symb], [py_symb], [ay_symb]])
    # Gx = Matrix(np.zeros((gx.shape[0], vectx.shape[0])))
    # Gx[:, 0] = diff(gx, vectx[0])
    # Gx[:, 1] = diff(gx, vectx[1])
    # Gx[:, 2] = diff(gx, vectx[2])
    # Gx[:, 3] = diff(gx, vectx[3])
    # Gx[:, 4] = diff(gx, vectx[4])
    # Gx[:, 5] = diff(gx, vectx[5])

    # Call the LaTeX printer for illustration purposes
    # init_session(quiet=True)
    # sympy.init_printing(use_latex=True)  # Use LaTeX for symbols
    # init_vprinting()  # Interactive LaTeX for dot symbols
    # dtd = sympy.symbols('Delta_t')
    #
    # # Dynamic transition matrix based on Newton equations of motion
    # fx = sympy.Matrix([px_symb + dtd * vx_symb + (dtd ** 2 / 2) * ax_symb,
    #                    vx_symb + dtd * ax_symb,
    #                    ax_symb,
    #                    py_symb + dtd * vy_symb + (dtd ** 2 / 2) * ay_symb,
    #                    vy_symb + dtd * ay_symb,
    #                    ay_symb])
    # Fx = fx.jacobian(vectx)

    # Measurement matrix (Jacobian in EKF)
    trips_ph = []
    for trip in trips:  # ['2020-05-15-US-MTV-1', '2021-04-29-US-MTV-2', '2021-04-21-US-MTV-2']
        phones = os.listdir(trip)
        # loop through phone folders
        for phone in phones:
            # check for valid folder and file
            # dt = 0.001 if phone == 'XiaomiMi8' else 0.001
            folder = join(trip, phone)
            if isfile(folder):
                continue
            trip_phone = trip + '/' + phone
            # print(trip_phone)
            # trips_ph.append(trip_phone)
            # print('\nfolders_phones: ', trips_ph)
            # print('\nfolders_phones: ', len(trips_ph))

            # Select the date/phone data only
            # phone = df_baseline['phone'][df_baseline['date'] == date].any()
            dfs = df_baseline.loc[(df_baseline['date'] == trip) & (df_baseline['phone'] == phone)]  # rtk_wls
            dfs.reset_index(inplace=True)

            # dfs_wls = df_wls.loc[(df_gt['date'] == trip) & (df_wls['phone'] == phone)]  # wls
            # dfs_wls.reset_index(inplace=True)

            dfs_gt = df_gt.loc[(df_gt['date'] == trip) & (df_gt['phone'] == phone)]  # ground truth
            dfs_gt.reset_index(inplace=True)
            print('\nProcessing:', trip + '/' + phone)

            xg, yg = proj.transform(wgs, bng, dfs['long'], dfs['lat'])  # GNSS data in x,y
            xgt, ygt = proj.transform(wgs, bng, dfs_gt['long'], dfs_gt['lat'])  # GNSS ground truth data in x,y

            # Reading the corresponding IMU accelerometer data
            dfx = pd.read_csv(csvpath_train(trip, phone, file),
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
            dis = len(dfo['ay'])
            # print('dfo:', dis)

            x_dim = 6
            z_dim = 4
            sigma_x = 5.0  # position SD m
            sigma_v = 0.6 if phone == 'XiaomiMi8' else 0.1  # velocity SD m/s
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
            end_df = len(Mrts_kf)
            # print('size of Mrts_kf: ', end_df)

            dfo['accl_long'], dfo['accl_lat'] = proj.transform(bng, wgs, ax, ay)  # KF output in long/lat
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
            dfo.to_csv(csvpath_train(trip, phone, newfile))  # Save output GNSS/IMU
            # print()

            # Ground truth, rtk_wls, kf, and rts kf, and wls solution
            llh_gt = dfs_gt[['lat', 'long']].to_numpy()
            llh_rtk_wls = dfs[['lat', 'long']].to_numpy()
            # llh_wls = dfs_wls[['lat', 'long']].to_numpy()
            llh_kf = dfo[['kf_lat', 'kf_long']].to_numpy()
            llh_rts = dfo[['rts_lat', 'rts_long']].to_numpy()
            print(llh_gt)

            # balanced_signal = llh_rtk_wls - np.mean(llh_rtk_wls)
            # spectrum = np.fft.rfft(balanced_signal)

            print('length of llh_gt:', len(llh_gt))
            print('length of llh_rtk_wls:', len(llh_rtk_wls))
            # print('length of llh_wls:', len(llh_wls))
            print('length of llh_kf:', len(llh_kf))
            print('length of llh_rts:', len(llh_rts))
            print()

            # Distance from ground truth
            vd_rtk_wls = vincenty_distance(llh_rtk_wls, llh_gt)
            # vd_wls = vincenty_distance(llh_wls, llh_gt)
            vd_kf = vincenty_distance(llh_kf, llh_gt)
            vd_rts = vincenty_distance(llh_rts, llh_gt)

            # Score
            score_bl_rtk = calc_score(llh_rtk_wls, llh_gt)
            # score_bl_wls = calc_score(llh_wls, llh_gt)
            score_kf = calc_score(llh_kf, llh_gt)
            score_rts = calc_score(llh_rts, llh_gt)

            print(f'Score rtk_wls: {score_bl_rtk:.4f} [m]')
            # print(f'Score wls:     {score_bl_wls:.4f} [m]')
            print(f'Score KF:      {score_kf:.4f} [m]')
            print(f'Score RTS KF:  {score_rts:.4f} [m]')

            # # Plot distance error
            plt.figure()
            plt.title('Distance error')
            plt.xlabel('Time [s]')
            plt.ylabel('Distance error [m]')
            # plt.plot(vd_rtk_wls, label=f'Score rtk : {score_bl_rtk:.4f} m')
            # plt.plot(vd_wls, label=f'WLS Baseline:     {score_bl_wls:.4f} m')
            # plt.plot(vd_kf, label=f'Score KF, Score:             {score_kf:.4f} m')
            plt.plot(vd_rts, label=f'MAP_RTK:        {score_rts:.4f} m')
            plt.legend()
            plt.grid()
            plt.ylim([0, 30])
            #
            # # Plot figures: X and T
            # plt.figure()
            # plt.plot(dfo['unixTime'], xgt, '-', label='groundTruth')
            # plt.plot(dfo['unixTime'], xg, label='gnss')
            # plt.plot(dfo['unixTime'], xkf, '-', label='KF fusion')
            # plt.plot(dfo['unixTime'], Mrts_kf[:, 0], '-', label='RTS KF')
            # plt.legend()
            #
            # # Plot figures: Y and T
            # plt.figure()
            # plt.plot(dfo['unixTime'], ygt, '-', label='groundTruth')
            # plt.plot(dfo['unixTime'], yg, label='gnss')
            # plt.plot(dfo['unixTime'], ykf, '-', label='KF fusion')
            # plt.plot(dfo['unixTime'], Mrts_kf[:, 3], '-', label='RTS KF')
            # plt.legend()
            #
            # # Plot figures:
            # plt.figure()
            # plt.plot(xgt, ygt, '-', label='groundTruth')
            # plt.plot(xg, yg, label='gnss')
            # plt.plot(xkf, ykf, '-', label='KF fusion')
            # plt.plot(Mrts_kf[:, 0], Mrts_kf[:, 3], '-', label='RTS KF')
            # plt.legend()
            # plt.show()

            # plt.plot(xgt, ygt, '-', label='groundTruth')
            # plt.plot(smooth(xg, yg, 19), label='gnss smooth')
            # plt.plot(smooth(xkf, ykf, 19), '-', label='KF fusion smooth')
            # plt.plot(smooth(Mrts_kf[:, 0], Mrts_kf[:, 3], 19), '-', label='RTS KF smooth')
            # plt.legend()
            # plt.show()

            # Plot figures:
            plt.figure()
            plt.plot(dfs_gt['long'], dfs_gt['lat'], '-', label='groundTruth')
            plt.plot(dfs['long'], dfs['lat'], label='gnss')
            plt.plot(dfo['kf_long'], dfo['kf_lat'], '-', label='KF fusion')
            plt.plot(dfo['rts_long'], dfo['rts_lat'], '-', label='RTS KF')
            plt.legend()
            plt.show()

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
