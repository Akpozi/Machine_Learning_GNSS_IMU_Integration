"""
RTKLIB-Py: run_ppk_multi_MLPathPredicy.py - convert raw android files to rinex and run PPK solutions for GDSC_2022
data set with RTKLIB and/or rtklib-py.

Copyright (c) 2022 Tim Everett (from rtklib_py)
Copyright (c) 2022 Akpo Siemuri
Date - 2022-05-23
Modified: 2022-08-11

NOTE: PPK solution is based on the Machine Learning predicted driving paths of the phone to take care of multipath errors
"""

import sys

if 'rtklib-py/src' not in sys.path:
    sys.path.append('rtklib-py/src')
if 'android_rinex/src' not in sys.path:
    sys.path.append('android_rinex/src')

import os
import shutil
from os.path import join, isdir, isfile
import subprocess
from python import gnsslogger_to_rnx as rnx

################################################################
import warnings

warnings.simplefilter('ignore')
from glob import glob
from time import time
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# ###############################################################

# set run parameters
maxepoch = None  # max number of epochs, used for debug, None = no limit
trace_level = 3  # debug trace level

# Set solution choices
ENABLE_PY = True  # Use RTKLIB-PY to generate solutions
ENABLE_RTKLIB = False  # Use RTKLIB to generate solutions
OVERWRITE_RINEX = False  # overwrite existing rinex filex
OVERWRITE_SOL = False  # overwrite existing solution files

# specify location of input folder and files
test_train = 'test'  # test_sample   train
datadir = '/home/akpo/GSDC_2022_rtklib_py/data' + '/' + test_train
# basefiles = '../*0.2*o'  # rinex2, use this for rtklib only
basefiles = '../base.obs'  # rinex3, use this for python only
navfiles = '../*MN.rnx'  # navigation files with wild cards

# Setup for RTKLIB
binpath_rtklib = '/home/akpo/GSDC_2022_rtklib_py/rtklib/demo5_b34f1/rnx2rtkp.exe'
cfgfile_rtklib = '/home/akpo/GSDC_2022_rtklib_py/config/ppk_phone_0510.conf'
soltag_rtklib = '_rtklib_combined_noreset_snr52'  # postfix for solution file names

# Setup for rtklib-py
cfgfile_highway = '/home/akpo/GSDC_2022_rtklib_py/config/ppk_phone_0510_highway.py'
cfgfile_treeway = '/home/akpo/GSDC_2022_rtklib_py/config/ppk_phone_0510_treeway.py'
cfgfile_downtown = '/home/akpo/GSDC_2022_rtklib_py/config/ppk_phone_0510_downtown.py'
soltag_py = '_py0510_combined_noreset_MLPathPredict'  # postfix for solution file names

# List dataset choices
PHONES = ['GooglePixel4', 'GooglePixel4XL', 'Pixel4Modded', 'GooglePixel5', 'GooglePixel6Pro', 'SamsungGalaxyS20Ultra',
          'XiaomiMi8']
BASE_POS = {'slac': [-2703115.9184, -4291767.2037, 3854247.9027],  # WGS84 XYZ coordinates
            'vdcy': [-2497836.5139, -4654543.2609, 3563028.9379],
            'p222': [-2689640.2891, -4290437.3671, 3865050.9313]}
# ################################################

# Implement ML model for driving path prediction: (HIGHWAY, TREELINE WAY or DOWNTOWN)
train_paths = '/home/akpo/GSDC_2022_rtklib_py/data/train_test_paths/rtk_wls2fix_hw_clock_error_rnx2rtkp_train_data.csv'
test_paths = '/home/akpo/GSDC_2022_rtklib_py/data/train_test_paths/cys_rtk_wls2fix_hw_clock_error_test11.csv'


def area_prediction():
    BASE_DIR = '/home/akpo/GSDC_2022_rtklib_py/data'
    train_base = pd.read_csv(train_paths)
    train_base[['collectionName', 'phoneName']] = train_base.tripId.str.split("/", expand=True)
    train_base.rename(
        columns={'UnixTimeMillis': 'millisSinceGpsEpoch', 'LatitudeDegrees': 'latDeg', 'LongitudeDegrees': 'lngDeg'},
        inplace=True)

    train_base = train_base.sort_values([
        "collectionName", "phoneName", "millisSinceGpsEpoch"
    ]).reset_index(drop=True)
    train_base['area'] = train_base['collectionName'].map(lambda x: x.split('-')[4])

    test_base = pd.read_csv(test_paths)
    test_base[['collectionName', 'phoneName']] = test_base.tripId.str.split("/", expand=True)
    test_base.rename(
        columns={'UnixTimeMillis': 'millisSinceGpsEpoch', 'LatitudeDegrees': 'latDeg', 'LongitudeDegrees': 'lngDeg'},
        inplace=True)
    test_base = test_base.sort_values([
        "collectionName", "phoneName", "millisSinceGpsEpoch"
    ]).reset_index(drop=True)
    test_base['area'] = test_base['collectionName'].map(lambda x: x.split('-')[4])

    train_name = np.array(sorted(path.split('/')[-1] for path in glob(f'{BASE_DIR}/train/*')))
    train_highway = train_name[np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
         31, 36, 37, 38, 39, 40, 41]) - 1]
    train_tree = train_name[
        np.array([43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]) - 1]
    train_downtown = train_name[np.array([32, 33, 34, 35, 42]) - 1]

    train_base['area_target'] = -1
    train_base.loc[train_base['collectionName'].isin(train_highway), 'area_target'] = 0
    train_base.loc[train_base['collectionName'].isin(train_tree), 'area_target'] = 1
    train_base.loc[train_base['collectionName'].isin(train_downtown), 'area_target'] = 2

    def processing_downtown(input_df: pd.DataFrame, is_train=False):
        output_df = input_df.groupby('collectionName')[['latDeg', 'lngDeg']].std()
        if is_train:
            output_df = output_df.merge(
                input_df.groupby('collectionName')[['area_target']].first(),
                on='collectionName')
        output_df = output_df.merge(
            input_df.groupby('collectionName')['area'].first(),
            on='collectionName')
        output_df = output_df.merge(
            input_df.groupby('collectionName')['phoneName'].unique().apply(list),
            on='collectionName')
        return output_df

    train = processing_downtown(train_base, is_train=True)
    train['downtown_target'] = (train['area_target'] == 2).astype(int)

    downtown_model_knn = KNeighborsClassifier(n_neighbors=1)
    downtown_model_knn.fit(
        train[['latDeg', 'lngDeg']],
        train['downtown_target'],
    )

    def processing_highway_tree(input_df: pd.DataFrame, is_train=False):
        output_df = input_df.groupby('collectionName')[['latDeg', 'lngDeg']].min()
        if is_train:
            output_df = output_df.merge(
                input_df.groupby('collectionName')[['area_target']].first(),
                on='collectionName')
        output_df = output_df.merge(
            input_df.groupby('collectionName')['area'].first(),
            on='collectionName')
        output_df = output_df.merge(
            input_df.groupby('collectionName')['phoneName'].unique().apply(list),
            on='collectionName')
        return output_df

    train = processing_highway_tree(train_base, is_train=True)

    highway_tree_model_knn = KNeighborsClassifier(n_neighbors=1)
    highway_tree_model_knn.fit(
        train.loc[train['area_target'] != 2, ['latDeg', 'lngDeg']],
        train.loc[train['area_target'] != 2, 'area_target'],
    )

    def predict_area(test_base):
        test_base = test_base.copy()
        test_base = test_base.sort_values([
            "collectionName", "phoneName", "millisSinceGpsEpoch"
        ]).reset_index(drop=True)
        test_base['area'] = test_base['collectionName'].map(lambda x: x.split('-')[4])

        test = processing_downtown(test_base)
        downtown_pred = downtown_model_knn.predict(test[['latDeg', 'lngDeg']])

        test = processing_highway_tree(test_base)
        test.loc[downtown_pred == 1, 'area_pred'] = 2
        pred = highway_tree_model_knn.predict(test.loc[test['area_pred'].isnull(), ['latDeg', 'lngDeg']])
        test.loc[test['area_pred'].isnull(), 'area_pred'] = pred
        test['area_pred'] = test['area_pred'].astype(int)
        test['collectionName'] = test.index

        test_highway = []
        test_tree = []
        test_downtown = []
        for collection, area_pred in test[['collectionName', 'area_pred']].itertuples(index=False):
            if area_pred == 0:
                test_highway.append(collection)
            elif area_pred == 1:
                test_tree.append(collection)
            else:
                test_downtown.append(collection)
        return test_highway, test_tree, test_downtown

    return train_highway, train_tree, train_downtown, predict_area(test_base)


TRAIN_HIGHWAY, TRAIN_TREEWAY, TRAIN_DOWNTOWN, \
(TEST_HIGHWAY, TEST_TREEWAY, TEST_DOWNTOWN) = area_prediction()

print()
print('####### Prediction results for datasets drive paths #######')
print('TEST_HIGHWAY: ', sorted(TEST_HIGHWAY))
print('TEST_TREEWAY: ', sorted(TEST_TREEWAY))
print('TEST_DOWNTOWN: ', sorted(TEST_DOWNTOWN))
print('####### End of drive paths prediction #######')
print()

print('Processing data based on predicted driving paths (HIGHWAY, TREELINE WAY or DOWNTOWN)')


# ###################################################################################################################


# input structure for rinex conversion
class args:
    def __init__(self):
        # Input parameters for conversion to rinex
        self.slip_mask = 0  # overwritten below
        self.fix_bias = True
        self.timeadj = 1e-7
        self.pseudorange_bias = 0
        self.filter_mode = 'sync'
        # Optional hader values for rinex files
        self.marker_name = ''
        self.observer = ''
        self.agency = ''
        self.receiver_number = ''
        self.receiver_type = ''
        self.receiver_version = ''
        self.antenna_number = ''
        self.antenna_type = ''


# get list of data sets in data path
datasets = sorted(os.listdir(datadir))


# ------------------------Process Highway Drive path -------------------------- #
def highway():
    # Copy and read config file
    if ENABLE_PY:
        shutil.copyfile(cfgfile_highway, '__ppk_config.py')  # __ppk_config_highway
        import __ppk_config as cfg
        import python.rtklib_py.src.rinex as rn
        import python.rtklib_py.src.rtkcmn as gn
        from python.rtklib_py.src.rtkpos import rtkinit
        from python.rtklib_py.src.postpos import procpos, savesol

    # function to convert single rinex file
    def convert_rnx(folder, rawFile, rovFile, slipMask):
        os.chdir(folder)
        argsIn = args()
        argsIn.input_log = rawFile
        argsIn.output = os.path.basename(rovFile)
        argsIn.slip_mask = slipMask
        rnx.convert2rnx(argsIn)

    # function to run single RTKLIB-Py solution
    def run_ppk(folder, rovfile, basefile, navfile, solfile):
        # init solution
        if trace_level > 0:
            trcfile = os.path.join(folder, 'gnss_log' + '.trace')
            sys.stderr = open(trcfile, "w")
        os.chdir(folder)
        print('This ff:', folder)

        gn.tracelevel(trace_level)
        nav = rtkinit(cfg)
        nav.maxepoch = maxepoch
        print('The phone being processed is: ', folder)

        # load rover obs
        rov = rn.rnx_decode(cfg)
        print('   Reading rover obs...')
        if nav.filtertype == 'backward':
            maxobs = None  # load all obs for backwards
        else:
            maxobs = maxepoch
        rov.decode_obsfile(nav, rovfile, maxobs)

        # load base obs and location
        base = rn.rnx_decode(cfg)
        print('   Reading base obs...')
        base.decode_obsfile(nav, basefile, None)

        # determine base location from original base obs file name
        if len(BASE_POS) > 1:
            baseName = glob('../*.2*o')[0][-12:-8]
            nav.rb[0:3] = BASE_POS[baseName]
        elif nav.rb[0] == 0:
            nav.rb = base.pos  # from obs file

        # load nav data from rover obs
        print('   Reading nav data...')
        rov.decode_nav(navfile, nav)

        # calculate solution
        print('   Calculating solution...')
        sol = procpos(nav, rov, base)
        print()

        # save solution to file
        savesol(sol, solfile)
        print()
        return rovfile

    # function to run single RTKLIB solution
    def run_rtklib(folder, rovfile, basefile, navfile, solfile):
        # create command to run solution
        rtkcmd = '%s -x 3 -y 2 -k %s -o %s %s %s %s' % \
                 (binpath_rtklib, cfgfile_rtklib, solfile, rovfile, basefile, navfile)

        # run command
        os.chdir(folder)
        subprocess.run(['wine', rtkcmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # ###### Start of main code ##########################

    def main_highway():
        # loop through data set folders
        rinexIn = []
        ppkIn = []
        rtklibIn = []
        for dataset in TEST_HIGHWAY:  # datasets
            for phone in PHONES:
                # skip if no folder for this phone
                folder = join(datadir, dataset, phone)
                if not isdir(folder):
                    continue
                os.chdir(folder)
                rawFile = join('supplemental', 'gnss_log.txt')
                rovFile = join('supplemental', 'gnss_log.obs')

                rinex = False
                # check if need rinex conversion
                if OVERWRITE_RINEX or not isfile(rovFile):
                    # generate list of input parameters for each rinex conversion
                    if phone == 'SamsungS20Ultra':  # Use cycle slip flags for Samsung phones
                        slipMask = 0  # 1 to unmask recevier cycle slips
                    else:
                        slipMask = 0
                    rinexIn.append((folder, rawFile, rovFile, slipMask))
                    print(rawFile, '->', rovFile)
                    rinex = True

                # check if need to create PPK solution
                try:
                    baseFile = glob(basefiles)[0]
                    navFile = glob(navfiles)[0]
                    solFile = rovFile[:-4] + soltag_py + '.pos'
                    solFile_rtklib = rovFile[:-4] + soltag_rtklib + '.pos'
                except:
                    print(folder, '  Error: Missing file')
                    continue
                if ENABLE_PY and (OVERWRITE_SOL == True or len(glob(solFile)) == 0
                                  or rinex == True):
                    # generate list of input/output files for each python ppk solution
                    # print('PY: ', join(dataset, phone))
                    ppkIn.append((folder, rovFile, baseFile, navFile, solFile))
                if ENABLE_RTKLIB and (OVERWRITE_SOL == True or
                                      len(glob(solFile_rtklib)) == 0 or rinex == True):
                    # generate list of input/output files for each rtklib ppk solution
                    print('RTKLIB: ', join(dataset, phone))
                    rtklibIn.append((folder, rovFile, baseFile, navFile, solFile_rtklib))

        if len(rinexIn) > 0:
            print('\nConverting raw GNSS android files to rinex files...')
            # generate rinx obs files in parallel, does not give error messages
            # with Pool() as pool: # defaults to using cpu_count for number of procceses
            #    res = pool.starmap(convert_rnx, rinexIn)
            # run sequentially, use for debug
            for input in rinexIn:
                convert_rnx(input[0], input[1], input[2], input[3])

        if ENABLE_PY and len(ppkIn) > 0:
            print('\nCalculate PPK solutions...')
            # run PPK solutions in parallel, does not give error messages
            # with Pool() as pool: # defaults to using cpu_count for number of procceses
            #     res = pool.starmap(run_ppk, ppkIn)
            # run sequentially, use for debug
            for input in ppkIn:
                print('Using HIGHWAY Configuration for drive path: ', input[0][41:60])
                # print('This is the input', input[0])
                run_ppk(input[0], input[1], input[2], input[3], input[4])

        if ENABLE_RTKLIB and len(rtklibIn) > 0:
            print('\nCalculate RTKLIB solutions...')
            # run PPK solutions in parallel, does not give error messages
            # with Pool() as pool: # defaults to using cpu_count for number of procceses
            #     res = pool.starmap(run_rtklib, rtklibIn)
            # run sequentially, use for debug
            for input in rtklibIn:
                run_rtklib(input[0], input[1], input[2], input[3], input[4])

    # Run main_highway():
    main_highway()


# ------------------------Process Treeline Drive path -------------------------- #
def treeway():
    # Copy and read config file
    if ENABLE_PY:
        # shutil.copyfile(cfgfile_treeway, '__ppk_config.py')  # __ppk_config_treeway
        default_file = '/home/akpo/GSDC_2022_rtklib_py/scr/__ppk_config.py'
        # open both configuration files
        with open(cfgfile_treeway, 'r') as firstfile, open(default_file, 'w') as secondfile:
            # read content from first file
            for line in firstfile:
                # write content to second file
                secondfile.write(line)
        import __ppk_config as cfg
        import python.rtklib_py.src.rinex as rn
        import python.rtklib_py.src.rtkcmn as gn
        from python.rtklib_py.src.rtkpos import rtkinit
        from python.rtklib_py.src.postpos import procpos, savesol

    # function to convert single rinex file
    def convert_rnx(folder, rawFile, rovFile, slipMask):
        os.chdir(folder)
        argsIn = args()
        argsIn.input_log = rawFile
        argsIn.output = os.path.basename(rovFile)
        argsIn.slip_mask = slipMask
        rnx.convert2rnx(argsIn)

    # function to run single RTKLIB-Py solution
    def run_ppk(folder, rovfile, basefile, navfile, solfile):
        # init solution
        if trace_level > 0:
            trcfile = os.path.join(folder, 'gnss_log' + '.trace')
            sys.stderr = open(trcfile, "w")
        os.chdir(folder)
        print('This ff:', folder)

        gn.tracelevel(trace_level)
        nav = rtkinit(cfg)
        nav.maxepoch = maxepoch
        print('The phone being processed is: ', folder)

        # load rover obs
        rov = rn.rnx_decode(cfg)
        print('   Reading rover obs...')
        if nav.filtertype == 'backward':
            maxobs = None  # load all obs for backwards
        else:
            maxobs = maxepoch
        rov.decode_obsfile(nav, rovfile, maxobs)

        # load base obs and location
        base = rn.rnx_decode(cfg)
        print('   Reading base obs...')
        base.decode_obsfile(nav, basefile, None)

        # determine base location from original base obs file name
        if len(BASE_POS) > 1:
            baseName = glob('../*.2*o')[0][-12:-8]
            nav.rb[0:3] = BASE_POS[baseName]
        elif nav.rb[0] == 0:
            nav.rb = base.pos  # from obs file

        # load nav data from rover obs
        print('   Reading nav data...')
        rov.decode_nav(navfile, nav)

        # calculate solution
        print('   Calculating solution...')
        sol = procpos(nav, rov, base)
        print()

        # save solution to file
        savesol(sol, solfile)
        print()
        return rovfile

    # function to run single RTKLIB solution
    def run_rtklib(folder, rovfile, basefile, navfile, solfile):
        # create command to run solution
        rtkcmd = '%s -x 3 -y 2 -k %s -o %s %s %s %s' % \
                 (binpath_rtklib, cfgfile_rtklib, solfile, rovfile, basefile, navfile)

        # run command
        os.chdir(folder)
        subprocess.run(['wine', rtkcmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # ###### Start of main code ##########################

    def main_treeway():
        # loop through data set folders
        rinexIn = []
        ppkIn = []
        rtklibIn = []
        for dataset in TEST_TREEWAY:  # datasets
            for phone in PHONES:
                # skip if no folder for this phone
                folder = join(datadir, dataset, phone)
                if not isdir(folder):
                    continue
                os.chdir(folder)
                rawFile = join('supplemental', 'gnss_log.txt')
                rovFile = join('supplemental', 'gnss_log.obs')

                rinex = False
                # check if need rinex conversion
                if OVERWRITE_RINEX or not isfile(rovFile):
                    # generate list of input parameters for each rinex conversion
                    if phone == 'SamsungS20Ultra':  # Use cycle slip flags for Samsung phones
                        slipMask = 0  # 1 to unmask recevier cycle slips
                    else:
                        slipMask = 0
                    rinexIn.append((folder, rawFile, rovFile, slipMask))
                    print(rawFile, '->', rovFile)
                    rinex = True

                # check if need to create PPK solution
                try:
                    baseFile = glob(basefiles)[0]
                    navFile = glob(navfiles)[0]
                    solFile = rovFile[:-4] + soltag_py + '.pos'
                    solFile_rtklib = rovFile[:-4] + soltag_rtklib + '.pos'
                except:
                    print(folder, '  Error: Missing file')
                    continue
                if ENABLE_PY and (OVERWRITE_SOL == True or len(glob(solFile)) == 0
                                  or rinex == True):
                    # generate list of input/output files for each python ppk solution
                    # print('PY: ', join(dataset, phone))
                    ppkIn.append((folder, rovFile, baseFile, navFile, solFile))
                if ENABLE_RTKLIB and (OVERWRITE_SOL == True or
                                      len(glob(solFile_rtklib)) == 0 or rinex == True):
                    # generate list of input/output files for each rtklib ppk solution
                    print('RTKLIB: ', join(dataset, phone))
                    rtklibIn.append((folder, rovFile, baseFile, navFile, solFile_rtklib))

        if len(rinexIn) > 0:
            print('\nConverting raw GNSS android files to rinex files...')
            # generate rinx obs files in parallel, does not give error messages
            # with Pool() as pool: # defaults to using cpu_count for number of procceses
            #    res = pool.starmap(convert_rnx, rinexIn)
            # run sequentially, use for debug
            for input in rinexIn:
                convert_rnx(input[0], input[1], input[2], input[3])

        if ENABLE_PY and len(ppkIn) > 0:
            print('\nCalculate PPK solutions...')
            # run PPK solutions in parallel, does not give error messages
            # with Pool() as pool: # defaults to using cpu_count for number of procceses
            #     res = pool.starmap(run_ppk, ppkIn)
            # run sequentially, use for debug
            for input in ppkIn:
                print('Using TREEWAY Configuration for drive path: ', input[0][41:60])
                # print('This is the input', input[0])
                run_ppk(input[0], input[1], input[2], input[3], input[4])

        if ENABLE_RTKLIB and len(rtklibIn) > 0:
            print('\nCalculate RTKLIB solutions...')
            # run PPK solutions in parallel, does not give error messages
            # with Pool() as pool: # defaults to using cpu_count for number of procceses
            #     res = pool.starmap(run_rtklib, rtklibIn)
            # run sequentially, use for debug
            for input in rtklibIn:
                run_rtklib(input[0], input[1], input[2], input[3], input[4])

    # Run main_treeway():
    main_treeway()


# ------------------------Process Downtown Drive path -------------------------- #
def downtown():
    # Copy and read config file
    if ENABLE_PY:
        # shutil.copyfile(cfgfile_downtown, '__ppk_config.py')  # __ppk_config_treeway
        default_file = '/home/akpo/GSDC_2022_rtklib_py/scr/__ppk_config.py'
        # open both configuration files
        with open(cfgfile_downtown, 'r') as firstfile, open(default_file, 'w') as secondfile:
            # read content from first file
            for line in firstfile:
                # write content to second file
                secondfile.write(line)
        import __ppk_config as cfg
        import python.rtklib_py.src.rinex as rn
        import python.rtklib_py.src.rtkcmn as gn
        from python.rtklib_py.src.rtkpos import rtkinit
        from python.rtklib_py.src.postpos import procpos, savesol

    # function to convert single rinex file
    def convert_rnx(folder, rawFile, rovFile, slipMask):
        os.chdir(folder)
        argsIn = args()
        argsIn.input_log = rawFile
        argsIn.output = os.path.basename(rovFile)
        argsIn.slip_mask = slipMask
        rnx.convert2rnx(argsIn)

    # function to run single RTKLIB-Py solution
    def run_ppk(folder, rovfile, basefile, navfile, solfile):
        # init solution
        if trace_level > 0:
            trcfile = os.path.join(folder, 'gnss_log' + '.trace')
            sys.stderr = open(trcfile, "w")
        os.chdir(folder)
        print('This ff:', folder)

        gn.tracelevel(trace_level)
        nav = rtkinit(cfg)
        nav.maxepoch = maxepoch
        print('The phone being processed is: ', folder)

        # load rover obs
        rov = rn.rnx_decode(cfg)
        print('   Reading rover obs...')
        if nav.filtertype == 'backward':
            maxobs = None  # load all obs for backwards
        else:
            maxobs = maxepoch
        rov.decode_obsfile(nav, rovfile, maxobs)

        # load base obs and location
        base = rn.rnx_decode(cfg)
        print('   Reading base obs...')
        base.decode_obsfile(nav, basefile, None)

        # determine base location from original base obs file name
        if len(BASE_POS) > 1:
            baseName = glob('../*.2*o')[0][-12:-8]
            nav.rb[0:3] = BASE_POS[baseName]
        elif nav.rb[0] == 0:
            nav.rb = base.pos  # from obs file

        # load nav data from rover obs
        print('   Reading nav data...')
        rov.decode_nav(navfile, nav)

        # calculate solution
        print('   Calculating solution...')
        sol = procpos(nav, rov, base)
        print()

        # save solution to file
        savesol(sol, solfile)
        print()
        return rovfile

    # function to run single RTKLIB solution
    def run_rtklib(folder, rovfile, basefile, navfile, solfile):
        # create command to run solution
        rtkcmd = '%s -x 3 -y 2 -k %s -o %s %s %s %s' % \
                 (binpath_rtklib, cfgfile_rtklib, solfile, rovfile, basefile, navfile)

        # run command
        os.chdir(folder)
        subprocess.run(['wine', rtkcmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # ###### Start of main code ##########################

    def main_downtown():
        # loop through data set folders
        rinexIn = []
        ppkIn = []
        rtklibIn = []
        for dataset in TEST_DOWNTOWN:  # datasets
            for phone in PHONES:
                # skip if no folder for this phone
                folder = join(datadir, dataset, phone)
                if not isdir(folder):
                    continue
                os.chdir(folder)
                rawFile = join('supplemental', 'gnss_log.txt')
                rovFile = join('supplemental', 'gnss_log.obs')

                rinex = False
                # check if need rinex conversion
                if OVERWRITE_RINEX or not isfile(rovFile):
                    # generate list of input parameters for each rinex conversion
                    if phone == 'SamsungS20Ultra':  # Use cycle slip flags for Samsung phones
                        slipMask = 0  # 1 to unmask recevier cycle slips
                    else:
                        slipMask = 0
                    rinexIn.append((folder, rawFile, rovFile, slipMask))
                    print(rawFile, '->', rovFile)
                    rinex = True

                # check if need to create PPK solution
                try:
                    baseFile = glob(basefiles)[0]
                    navFile = glob(navfiles)[0]
                    solFile = rovFile[:-4] + soltag_py + '.pos'
                    solFile_rtklib = rovFile[:-4] + soltag_rtklib + '.pos'
                except:
                    print(folder, '  Error: Missing file')
                    continue
                if ENABLE_PY and (OVERWRITE_SOL == True or len(glob(solFile)) == 0
                                  or rinex == True):
                    # generate list of input/output files for each python ppk solution
                    # print('PY: ', join(dataset, phone))
                    ppkIn.append((folder, rovFile, baseFile, navFile, solFile))
                if ENABLE_RTKLIB and (OVERWRITE_SOL == True or
                                      len(glob(solFile_rtklib)) == 0 or rinex == True):
                    # generate list of input/output files for each rtklib ppk solution
                    print('RTKLIB: ', join(dataset, phone))
                    rtklibIn.append((folder, rovFile, baseFile, navFile, solFile_rtklib))

        if len(rinexIn) > 0:
            print('\nConverting raw GNSS android files to rinex files...')
            # generate rinx obs files in parallel, does not give error messages
            # with Pool() as pool: # defaults to using cpu_count for number of procceses
            #    res = pool.starmap(convert_rnx, rinexIn)
            # run sequentially, use for debug
            for input in rinexIn:
                convert_rnx(input[0], input[1], input[2], input[3])

        if ENABLE_PY and len(ppkIn) > 0:
            print('\nCalculate PPK solutions...')
            # run PPK solutions in parallel, does not give error messages
            # with Pool() as pool: # defaults to using cpu_count for number of procceses
            #     res = pool.starmap(run_ppk, ppkIn)
            # run sequentially, use for debug
            for input in ppkIn:
                print('Using DOWNTOWN Configuration for drive path: ', input[0][41:60])
                # print('This is the input', input[0])
                run_ppk(input[0], input[1], input[2], input[3], input[4])

        if ENABLE_RTKLIB and len(rtklibIn) > 0:
            print('\nCalculate RTKLIB solutions...')
            # run PPK solutions in parallel, does not give error messages
            # with Pool() as pool: # defaults to using cpu_count for number of procceses
            #     res = pool.starmap(run_rtklib, rtklibIn)
            # run sequentially, use for debug
            for input in rtklibIn:
                run_rtklib(input[0], input[1], input[2], input[3], input[4])

    # Run main_downtown():
    main_downtown()


if __name__ == '__main__':
    t0 = time()
    print('Processing Highway drive paths')
    highway()
    print('Processing Treeway drive paths')
    treeway()
    print('Processing Downtown drive paths')
    downtown()
    # main()
    print('Runtime in minutes=%.1f' % ((time() - t0) / 60) + ' minutes')
    print('Runtime in hours=%.1f' % ((time() - t0) / 3600) + ' hour(s)')
