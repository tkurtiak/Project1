Hey, so put convert_raw_data.py in the project1 folder that the class website has in the zip.
It should be next to the sample report and the rotplot.py files. This way it access the data correctly.

^^^^^^^^^^^!!!!!!!!!!!!!!!^^^^^^^^^^^^^^^^^^^!!!!!!!!!!!!!!!!^^^^^^^^^^^^^^^


It will run and integrate stuff, the code is commented, there's a bunch of print outs mostly for debugging.
It only works on one dataset and just on the gyro data because I couldn't find IMUparams.mat to calibrate the accelerometer data

If the matplotlib plots work on anyones machines, theres only a few lines of code left to do accelerometer data as well.

BTW nothing is low-pass or hi-pass filtered yet. It's just a start.

But we do have attitude data now woo.