"""
RTKLIB-Py: rinex2_to_rinex3.py - Convert rinex 2 obs files to rinex 3

Copyright (c) 2022 Tim Everett (from rtklib_py)
Copyright (c) 2022 Akpo Siemuri
Date - 2022-05-23
"""

import os
from os.path import join
import subprocess

# specify input folder and files
test_train = 'test'
datadir = '/home/akpo/GSDC_2022_rtklib_py/data' + '/' + test_train
obsfiles = '*0.2*o'  # base obs files with wild cards
sigmask = 'GL1C,GL5X,RL1C,EL1X,EL5X'

# Setup for RTKLIB
binpath = '/home/akpo/GSDC_2022_rtklib_py/rtklib/demo5_b34f1/convbin.exe'

# get list of data sets in data path
datasets = sorted(os.listdir(datadir))

for dataset in datasets:
    try:
        os.chdir(join(datadir, dataset))
    except:
        continue
    print(dataset)
    rtkcmd = '%s %s -r rinex -mask %s -od -os -o base.obs' % \
        (binpath, obsfiles, sigmask)
    subprocess.run(rtkcmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # subprocess.run(["wine", crx2rnx_bin, datadir + '/' + crx_files[1]])
