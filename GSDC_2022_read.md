# Machine_Learning_GNSS_IMU_Integration
Google Smartphone Decimeter Challenge 2022 codes

## Implementation:
A machine learning (ML) based adaptive positioning approach to estimate the positions of the smartphone by utilizing post-processed kinematic (PPK) precise positioning techniques to process the GNSS datasets. The ML model is used to predict the driving paths (highways, tree-lined streets, or downtown areas). Depending on the predicted driving path, the PPK technique computes the user position using different configuration settings.

This is adapted from "Getting started with rtklib-py": 
https://www.kaggle.com/code/timeverett/getting-started-with-rtklib-py

# Steps taken:

## Step 1: Retrieve base observation and satellite navigation files using
get_base_data.py

## Step 1.1: Converting base observation files from rinex2 to rinex3 format
### NOTE: this step is done when running rtklib_py
rnxV2_to_V3.py

## Step 2: Convert android phone's raw GNSS files to RINEX V3 and use ML to predict driving path (Highway, Treelined way, or Downtown) of phones then generate the PPK solution files according to the predicted driving paths to take care of multipaths
run_ppk_multi_MLPathPredict.py

## Step 3: Combine RTKLIB solutions into a single .csv file
- ### Create csv file PPK solution files using timestamps in reference file
    create_baseline_csv_from_pos.py

- ### Create csv file from all training set ground truth files
    create_groundtruth_csv.py

## Step 4: Filtering out RTKLIB solutions with hardware clock discontinuites
- ### count hardware clock discontinuities in raw logs
- ### Used to filter out PPK solutions with hardware clock discontinuites
   count_clock_errors.py

## Step 5: Run “merge_rtk_wls_2fix-hwclock_errors_test.py” to replace PPK solutions with hardware clock discontinuites with the WLS provided by GSDC. This also generates the submission file that can be submited to Kaggle without GNSS/IMU integration.
   merge_rtk_wls_2fix-hwclock_errors_test.py

## Step 5: use the .cvs solution file from Step 6 in "gnss_imu_fussion_test.py" to implement loosely coupled integration of GNSS/IMU.
   gnss_imu_fussion_test.py
   
## Merge all the GNSS/IMU interation from all phones to one final submission file
- ### Submit CSV file to Kaggle
   create_gnssIMU_KF.py
