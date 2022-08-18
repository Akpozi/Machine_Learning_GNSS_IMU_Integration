"""
count_clock_errors.py - count hardware clock discontinuities in raw logs
"""

import os
from os.path import join, isfile

# ########## Input parameters ###############################

DATA_SET = 'test'
datapath = r'/media/akpo/akpo_LaCie/GSDC/GSDC_Codes_Data_backup_2022/GSDC_2022/data'

############################################################


# get list of data sets in data path
os.chdir(join(datapath, DATA_SET))
trips = os.listdir()

# loop through data set folders
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
        trip_phone = trip + '_' + phone

        infile = join(folder, 'supplemental', 'gnss_log.txt')

        # parse solution file
        clks, secs = [], []
        fid = open(infile, 'r')
        lines = fid.readlines(10000000)
        fid.close()
        for line in lines:
            x = line.split(',')
            if x[0] == 'Raw':
                clks.append(int(x[10]))
                secs.append(int(x[1]))

        dclks = clks[-1] - clks[0]
        dsecs = (secs[-1] - secs[0]) / 1000

        if dclks > 1:
            print('%3d/%.0f: %s' % (dclks, dsecs, trip_phone))
