Project1.py is a python script which reads IMU data and calculates attitude based on 3 different methods:
1. Gyroscope Integration
2. Accelerometer
3. Madgwick Filter

The data is then plotted for user review.  

Data must be stored in the following folder structure:
IMU Data:   Data/Train/IMU/filename
VICON Data: Data/Train/VICON/filename2
IMUParams.mat must be provided in the root directory

filename and filename2 are declared in the beginning of the code.

matplotlib, scipy, and numpy must be installed in order for this code to run.  Additionally tf from ROS directories are used in transforming between quaternion and Euler angles.

This code was written by:
Illya Semenov
Tim Kurtiak
Nathan Witztum 

Written for sumbission in ENAE 788M Project1a on 9/10/2019 at the University of Maryland.