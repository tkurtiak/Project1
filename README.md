Hey, so put convert_raw_data.py in the project1 folder that the class website has in the zip.
It should be next to the sample report and the rotplot.py files. This way it access the data correctly.

^^^^^^^^^^^!!!!!!!!!!!!!!!^^^^^^^^^^^^^^^^^^^!!!!!!!!!!!!!!!!^^^^^^^^^^^^^^^


It will run and integrate stuff, the code is commented, there's a bunch of print outs mostly for debugging.
It only works on one dataset and just on the gyro data because I couldn't find IMUparams.mat to calibrate the accelerometer data

If the matplotlib plots work on anyones machines, theres only a few lines of code left to do accelerometer data as well.

BTW nothing is low-pass or hi-pass filtered yet. It's just a start.

But we do have attitude data now woo.
-------------------------------------------------------------------------
%tkurtiak, 9/2/19
Hey - I added accelerometer data read in and added the resulting angles to the plots.  The accelerometer data seems to not capture the majority of the motions that the gyro does.  Also, it has a steady offset from the rest of the data so we may need to calibrate the starting angle or something. 
I'm also skeptical of the Psi and Phi data just after timestamp 20.  They might be hitting a singularity/gimbal lock.
