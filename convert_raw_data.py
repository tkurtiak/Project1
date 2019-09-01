from scipy import io
import matplotlib.pyplot as plt
import numpy as np 
file = io.loadmat("Data/Train/IMU/imuRaw1.mat")
VICONfile = io.loadmat("Data/Train/Vicon/viconRot1.mat")
raw = file['vals']
ts = file['ts']

#How to retrive data:   raw[which dtyp (ax,ay,az,wz,wx,wy), index of var]
#print(ts.shape) 
#print(raw.shape)

w_raw = np.array([raw[4,:],raw[5,:],raw[3,:]]) #reorder for x, y, z
#print(w_raw.shape)

# Calculate Bias as the average of the first 300 gyro datapoints when the gyro is at rest (assumed)
bias_w = np.array([w_raw[0,0:300].mean(),w_raw[1,0:300].mean(),w_raw[2,0:300].mean()])
#print(bias_w)
#print(bias_w.shape)

#converting w to rads
w_rad_x = (3300/1023)* (np.pi/180) * 0.3 * (w_raw[0,:]-bias_w[0])
w_rad_y = (3300/1023)* (np.pi/180) * 0.3 * (w_raw[1,:]-bias_w[1])
w_rad_z = (3300/1023)* (np.pi/180) * 0.3 * (w_raw[2,:]-bias_w[2])
w_rad = np.array([w_rad_x,w_rad_y,w_rad_z])

#erase the components, no need to have them hogging memory
#del w_rad_x
#del w_rad_y
#del w_rad_z

#print(w_rad)
#print(w_rad.shape)

#state vector is x: phi theta psi Euler angles (roll,pitch,yaw)
x_gyro_rad=np.zeros(w_rad.shape)

for i in range(x_gyro_rad.shape[1]):
	#print(w_rad[:,:i])
	#print(ts[0,:i])
	#print(i)
	temp=np.trapz(w_rad[:,:i],axis=1,x=ts[0,:i])
	#print(temp)
	x_gyro_rad[:,i] = np.transpose(temp)

print('done')	

#x_rad = np.trapz(w_rad,axis=1,x=ts)


#print(x_gyro_rad)


# Now Process VICON data
V_ts= VICONfile['ts']
V_rot_m= VICONfile['rots']


V_x_rad=np.zeros((3,V_rot_m.shape[2]))

#print(V_rot_m.shape)


for i in range(V_rot_m.shape[2]):
	R=V_rot_m[:,:,i]

	th1= -np.arcsin(R[2,0])
	th2= np.pi-th1

	psi1= np.arctan2(np.sign(np.cos(th1))*R[2,1],np.sign(np.cos(th1))*R[2,2])
	psi2= np.arctan2(np.sign(np.cos(th2))*R[2,1],np.sign(np.cos(th2))*R[2,2])

	phi1= np.arctan2(np.sign(np.cos(th1))*R[1,0],np.sign(np.cos(th1))*R[0,0])
	phi2= np.arctan2(np.sign(np.cos(th2))*R[1,0],np.sign(np.cos(th2))*R[0,0])


# Check this, Z-Y-X Euler Angle Assignments
	V_x_rad[0,i]=psi1
	V_x_rad[1,i]=th1
	V_x_rad[2,i]=phi1

#print(V_x_rad)


print('plotting')
#print(np.transpose(V_ts[:]).shape)
#print(V_x_rad[0,:].shape)

fig = plt.figure()
ax1 = fig.add_subplot(3,1,1,title = 'Psi')
#ax1.title('Psi')
ax1.plot(np.transpose(V_ts),V_x_rad[0,:],label = 'ViconPsi')
plt.hold(True)
ax1.plot(np.transpose(ts),x_gyro_rad[0,:],label = 'GyroPsi')
plt.legend()

ax2 = fig.add_subplot(3,1,2,title = 'Theta')
ax2.plot(np.transpose(V_ts),V_x_rad[1,:],label = 'ViconTheta')
plt.hold(True)
ax2.plot(np.transpose(ts),x_gyro_rad[1,:],label = 'GyroTheta')
plt.legend()

ax3 = fig.add_subplot(3,1,3,title = 'Phi')
ax3.plot(np.transpose(V_ts),V_x_rad[2,:],label = 'ViconPhi')
plt.hold(True)
ax3.plot(np.transpose(ts),x_gyro_rad[2,:],label = 'GyroPhi')
plt.legend()
plt.show()

raw_input('Press Enter to exit')
