"""
RTKLIB-Py: get_base_data.py - retrieve base observation and navigation data for the 2022 GSDC competition
Copyright (c) 2022 Tim Everett (from rtklib_py)
Copyright (c) 2022 Akpo Siemuri
Date - 2022-05-23
"""

# ################## import relevant modules ####################
import os
from datetime import datetime
import requests
import gzip
from glob import glob
import time
import subprocess

# ---------------------------------------------------------------------------------------------------------
""" 
"""
# Input parameters

test_train = 'train'
datadir = '/home/akpo/GSDC_2022_rtklib_py/data' + '/' + test_train
stas = ['slac', 'vdcy', 'p222']  # Bay Area, LA. backup for Bay Area
obs_url_base = 'https://geodesy.noaa.gov/corsdata/rinex'
nav_url_base = 'https://data.unavco.org/archive/gnss/rinex3/nav'
nav_file_base = 'AC0300USA_R_'  # 20210060000_01D_MN.rnx.gz

# Make sure you have downloaded this executable before running this code
crx2rnx_bin = '/home/akpo/GSDC_2022_rtklib_py/rtklib/demo5_b34f1/crx2rnx.exe'


def get_base_data():
    os.chdir(datadir)

    for dataset in sorted(os.listdir()):
        if not os.path.isdir(dataset):
            continue
        print(dataset)
        ymd = dataset.split('-')
        doy = datetime(int(ymd[0]), int(ymd[1]), int(ymd[2])).timetuple().tm_yday  # get day of year
        doy = str(doy).zfill(3)

        if len(glob(os.path.join(dataset, '*.*o'))) == 0:
            # get obs data
            i = 1 if '-LAX-' in dataset else 0  # use different base for LA
            fname = stas[i] + doy + '0.' + ymd[0][2:4] + 'd.gz'
            url = '/'.join([obs_url_base, ymd[0], doy, stas[i], fname])
            try:
                obs = gzip.decompress(requests.get(url).content)  # get obs and decompress
                # write obs data
                open(os.path.join(dataset, fname[:-3]), "wb").write(obs)
            except:
                # try backup CORS station
                i += 2
                fname = stas[i] + doy + '0.' + ymd[0][2:4] + 'd.gz'
                url = '/'.join([obs_url_base, ymd[0], doy, stas[i], fname])
                try:
                    obs = gzip.decompress(requests.get(url).content)  # get obs and decompress
                    # write obs data
                    open(os.path.join(dataset, fname[:-3]), "wb").write(obs)
                except:
                    print('Fail obs: %s' % dataset)

            # convert crx to rnx
            crx_files = glob(os.path.join(dataset, '*.*d'))
            ffname = stas[i] + doy + '0.' + ymd[0][2:4] + 'o'
            filepath = os.path.join(dataset, ffname)
            # calculate solution
            if len(crx_files) > 0:
                if os.path.exists(filepath):
                    print(filepath, ' Already exist skipping "convert crx to rnx"\n')
                else:
                    subprocess.run(["wine", crx2rnx_bin, datadir + '/' + crx_files[0]])
                    # os.system(crx2rnx_bin + ' ' + crx_files[0])

        # get nav data
        if len(glob(os.path.join(dataset, '*.rnx'))) > 0:
            continue  # file already exists
        fname = nav_file_base + ymd[0] + doy + '0000_01D_MN' + '.rnx.gz'
        url = '/'.join([nav_url_base, ymd[0], doy, fname])
        try:
            obs = gzip.decompress(requests.get(url).content)  # get obs and decompress
            # write nav data
            open(os.path.join(dataset, fname[:-3]), "wb").write(obs)
        except:
            print('Fail nav: %s' % dataset)


# This part is the first to execute when script is ran. It times the execution time of the function
start = time.time()  # StartTime
print("Script started at " + str(start) + " seconds")
get_base_data()  # Run the function to Download correctional files
end = time.time()  # EndTime
print("Script ended at " + str(end) + " seconds")
print("Script completed in " + str((end - start)) + " seconds OR " + str((end - start)/60) + " minutes")

print(' ')
print('Done! Download completed')
