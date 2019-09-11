from scipy import io
from scipy import interpolate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import numpy as np 
from rotplot import rotplot
#import tf

#Change file name here to view various data sets
filename = "imuRaw5"
filename2 = "viconRot5"

#load data
print('loading data')
file = io.loadmat("Data/Train/IMU/"+filename+".mat")
VICONfile = io.loadmat("Data/Train/Vicon/"+filename2+".mat")
IMUparams = io.loadmat("IMUParams.mat")
raw = file['vals']
ts = file['ts']
IMUparams_raw = IMUparams['IMUParams']
w_raw = np.array([raw[4,:],raw[5,:],raw[3,:]]) #reorder for x, y, z
a_raw = np.array([raw[0,:],raw[1,:],raw[2,:]]) 

# Find bias:
# assume bias is average of first 200 points
bias_w = np.array([w_raw[0,0:200].mean(),w_raw[1,0:200].mean(),w_raw[2,0:200].mean()])
# extract accelerometer bias from IMU params.  b_ax, b_ay, b_az
bias_a = np.array([IMUparams_raw[1,0],IMUparams_raw[1,1],IMUparams_raw[1,2]])
# extract accelerometer scale from IMU params.  s_x, s_y, s_z
scale_a = np.array([IMUparams_raw[0,0],IMUparams_raw[0,1],IMUparams_raw[0,2]])

#converting w to rads and account for bias
w_rad_x = (3300/1023)* (np.pi/180) * 0.3 * (w_raw[0,:]-bias_w[0])
w_rad_y = (3300/1023)* (np.pi/180) * 0.3 * (w_raw[1,:]-bias_w[1])
w_rad_z = (3300/1023)* (np.pi/180) * 0.3 * (w_raw[2,:]-bias_w[2])
w_rad = np.array([w_rad_x,w_rad_y,w_rad_z])

# convert accelerations to m/s2 account for bias
a_x4 = (a_raw[0,:]*scale_a[0]+bias_a[0])*9.81
a_y4 = (a_raw[1,:]*scale_a[1]+bias_a[1])*9.81
a_z4 = (a_raw[2,:]*scale_a[2]+bias_a[2])*9.81
a_scaled = np.array([a_x4,a_y4,a_z4])


#find Euler angles of Gyro only
x_gyro_rad=np.zeros(w_rad.shape)
for i in range(x_gyro_rad.shape[1]-1):
	#have to convert gyro [p q r] to euler angle dots:	
	
	#current conditions
	phi=x_gyro_rad[0,i]
	theta=x_gyro_rad[1,i]
	psi=x_gyro_rad[2,i]

	#conversion matrix
	#page 1 of stengel.mycpanel.princeton.edu/Quaternions.pdf
	LBI= np.array([[1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],[0, np.cos(phi), -np.sin(phi)],[0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]])
	

	#dEul= phidot,thetadot,psidot
	#w_rad[:,:i]= p q r

	#convert to d/dt Euler angle 
	dEul= np.matmul(LBI,w_rad[:,i])
	#integrate timestep
	dt=ts[0,i+1]-ts[0,i]
	temp= np.array([phi+ dt*dEul[0],theta+ dt*dEul[1],psi+ dt*dEul[2]])
	x_gyro_rad[:,i+1] = np.transpose(temp)



#find Euler angles of Accelerometer only
a_orientation = np.zeros(a_scaled.shape)
for i in range(x_gyro_rad.shape[1]):
	
	#slide 9
	a_phi = np.arctan2(a_y4[i],np.sqrt((a_x4[i]**2)+(a_z4[i]**2)))
	a_theta = np.arctan2(-a_x4[i],np.sqrt((a_y4[i]**2)+(a_z4[i]**2)))
	a_psi = np.arctan2(np.sqrt((a_x4[i]**2)+(a_y4[i]**2)),a_z4[i])

	a_orientation_temp = np.array([a_phi,a_theta,a_psi])
	a_orientation[:,i] = a_orientation_temp


# Process VICON data
V_ts= VICONfile['ts']
V_rot_m= VICONfile['rots']


V_x_rad=np.zeros((3,V_rot_m.shape[2]))
for i in range(V_rot_m.shape[2]):
	#R is the rotation matrix
	R=V_rot_m[:,:,i]

	# two solutions exist for a given matrix 
	th1= -np.arcsin(R[2,0])
	th2= np.pi-th1

	psi1= np.arctan2(np.sign(np.cos(th1))*R[2,1],np.sign(np.cos(th1))*R[2,2])
	psi2= np.arctan2(np.sign(np.cos(th2))*R[2,1],np.sign(np.cos(th2))*R[2,2])

	phi1= np.arctan2(np.sign(np.cos(th1))*R[1,0],np.sign(np.cos(th1))*R[0,0])
	phi2= np.arctan2(np.sign(np.cos(th2))*R[1,0],np.sign(np.cos(th2))*R[0,0])

	#take first solution since lack of information
	V_x_rad[0,i]=phi1
	V_x_rad[1,i]=th1
	V_x_rad[2,i]=psi1



#ON TO THE FILTER
print('starting filter')
######################################################################
#####TUNE Parameters############
Beta=.1
Muu=.1 #Muu accel + (1-Muu)*gyro
#####################################################################
#####################################################################





Magwick_x= np.zeros([3,x_gyro_rad.shape[1]])
est_quat= np.zeros([4,x_gyro_rad.shape[1]])
est_quat[0,0]=1 #let initial rotation be null rotation

qdot_est= np.zeros([4,x_gyro_rad.shape[1]])
qdot_est[0,0]=0

#cant go all the way to end since putting in state at t+1 every time
for i in range(x_gyro_rad.shape[1]-1):
	#current state
	q_est_t=est_quat[:,i]
	

	#GYRO BRANCH
	#slide 17 lecture 2b
	# right side of equation
	tempquat = [ 0, w_rad[0,i+1],w_rad[1,i+1],w_rad[2,i+1]]
	# left side
	temp = q_est_t
	
	#quaternion multiply
	a = temp
	b = tempquat
	orien_incr_quat = [0,0,0,0]
	orien_incr_quat[0] = .5*(a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3])
	orien_incr_quat[1] = .5*(a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2] )
	orien_incr_quat[2] = .5*(a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1] )
	orien_incr_quat[3] = .5*(a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0] )

	orientation_inc= np.array([orien_incr_quat[0],orien_incr_quat[1],orien_incr_quat[2],orien_incr_quat[3]])

	#ACCEL BRANCH
	
	#for simplicity of eq writing:
	q1=q_est_t[0]
	q2=q_est_t[1]
	q3=q_est_t[2]
	q4=q_est_t[3]

	#normalize a and take components
	ahat=a_scaled[:,i+1]*(1/np.linalg.norm(a_scaled[:,i+1]))
	ax=ahat[0]
	ay=ahat[1]
	az=ahat[2]
	#eq in slide 19,20
	little_f= np.array([(2*(q2*q4-q1*q3)-ax),(2*(q1*q2+q3*q4)-ay),(2*(0.5 - q2**2 - q3**2)-az)])
	big_J= np.array([[-2*q3, 2*q4, -2*q1, 2*q2],[2*q2, 2*q1, 2*q4, 2*q3],[0, -4*q2, -4*q3,0]])
	Del_F= np.matmul(np.transpose(big_J),little_f)
	#another mistake in slides, normalized by its own magnitude not little f
	grad_func= -Beta/(np.linalg.norm(Del_F)) * Del_F
	
	

	#BRANCH COMBINATION
	#INTEGRATE
	dt= ts[0,i+1]-ts[0,i]
	
	#temp2 is our next quaternion state
	temp2= q_est_t + (1-Muu)*dt*orientation_inc + Muu*grad_func
	#normalize before saving
	if np.count_nonzero(temp2)==0:
		temp2[0]=1
	temp2=temp2/(np.linalg.norm(temp2))
	#save
	est_quat[:,i+1] = np.transpose(temp2)
	
	#convert to Euler angles for plots
	#phi
	Magwick_x[2,i+1]= np.arctan2((2*(temp2[0]*temp2[1] + temp2[2]*temp2[3])),(1-2*(temp2[1]**2 + temp2[2]**2)))
	#theta
	Magwick_x[1,i+1]= np.arcsin(2*(temp2[0]*temp2[2] - temp2[3]*temp2[1]))
	#psi
	Magwick_x[0,i+1]= np.arctan2((2*(temp2[0]*temp2[3] + temp2[1]*temp2[2])),(1-2*(temp2[2]**2 + temp2[3]**2)))




print('plotting')

plot_time = np.zeros([1,x_gyro_rad.shape[1]])
plot_time_V = np.zeros([1,V_x_rad.shape[1]])
for i in range(x_gyro_rad.shape[1]):
	plot_time[0,i] = (ts[0,i]-ts[0,0])/1
for i in range(V_x_rad.shape[1]):	
	plot_time_V[0,i] = (V_ts[0,i]-V_ts[0,0])/1

plot_time = np.transpose(plot_time)
plot_time_V = np.transpose(plot_time_V)

fig = plt.figure(figsize=(12,10),dpi=80,facecolor = 'w', edgecolor = 'k')
ax1 = fig.add_subplot(3,1,1,title = 'Yaw, Psi')

ax1.plot(plot_time_V,V_x_rad[0,:],label = 'Vicon')
plt.hold(True)
ax1.plot(plot_time,x_gyro_rad[2,:],label = 'Gyro')
ax1.plot(plot_time,a_orientation[2,:],label = 'Accel')
ax1.plot(plot_time,Magwick_x[0,:],label = 'Magwick')

plt.ylabel('angle, Radians')
plt.legend(loc = 'lower right')

ax2 = fig.add_subplot(3,1,2,title = 'Pitch, Theta')
ax2.plot(plot_time_V,V_x_rad[1,:],label = 'Vicon')
plt.hold(True)
ax2.plot(plot_time,x_gyro_rad[1,:],label = 'Gyro')
ax2.plot(plot_time,a_orientation[1,:],label = 'Accel')
ax2.plot(plot_time,Magwick_x[1,:],label = 'Magwick')

plt.ylabel('angle, Radians')
plt.legend(loc = 'lower right')

ax3 = fig.add_subplot(3,1,3,title = 'Roll, Phi')
ax3.plot(plot_time_V,V_x_rad[2,:],label = 'Vicon')
plt.hold(True)
ax3.plot(plot_time,x_gyro_rad[0,:],label = 'Gyro')
ax3.plot(plot_time,a_orientation[0,:],label = 'Accel')
ax3.plot(plot_time,Magwick_x[2,:],label = 'Magwick')
plt.xlabel('time, s')
plt.ylabel('angle, Radians')
plt.legend(loc = 'lower right')

plt.savefig(filename + ".png")
plt.show()

# begin video production
print('Creating video simulation')

# produce ZYX Euler rotation matrices for every time step
# gyro data
print('The dimensions of x_gyro_rad are: ' + str(x_gyro_rad.shape))
Rzyx_gyro = np.zeros(shape = (3, 3, x_gyro_rad.shape[1]))
for i in range(0, Rzyx_gyro.shape[2]):
	Rz_gyro = np.array([[np.cos(x_gyro_rad[2, i]), -np.sin(x_gyro_rad[2, i]), 0],
		[np.sin(x_gyro_rad[2, i]), np.cos(x_gyro_rad[2, i]), 0],
		[0, 0, 1]])
	Ry_gyro = np.array([[np.cos(x_gyro_rad[1, i]), 0, np.sin(x_gyro_rad[1, i])],
		[0, 1, 0],
		[-np.sin(x_gyro_rad[1, i]), 0, np.cos(x_gyro_rad[1, i])]])
	Rx_gyro = np.array([[1, 0, 0],
		[0, np.cos(x_gyro_rad[0, i]), -np.sin(x_gyro_rad[0, i])],
		[0, np.sin(x_gyro_rad[0, i]), np.cos(x_gyro_rad[0, i])]])
	Rzyx_gyro[:, :, i] = np.matmul(Rz_gyro, np.matmul(Ry_gyro, Rx_gyro))

print('The dimensions of Rzyx_gyro are: ' + str(Rzyx_gyro.shape))

# accel data
print('The dimensions of a_orientation are: ' + str(a_orientation.shape))
Rzyx_accel = np.zeros(shape = (3, 3, a_orientation.shape[1]))
for i in range(0, Rzyx_accel.shape[2]):
	Rz_accel = np.array([[np.cos(a_orientation[2, i]), -np.sin(a_orientation[2, i]), 0],
		[np.sin(a_orientation[2, i]), np.cos(a_orientation[2, i]), 0],
		[0, 0, 1]])
	Ry_accel = np.array([[np.cos(a_orientation[1, i]), 0, np.sin(a_orientation[1, i])],
		[0, 1, 0],
		[-np.sin(a_orientation[1, i]), 0, np.cos(a_orientation[1, i])]])
	Rx_accel = np.array([[1, 0, 0],
		[0, np.cos(a_orientation[0, i]), -np.sin(a_orientation[0, i])],
		[0, np.sin(a_orientation[0, i]), np.cos(a_orientation[0, i])]])
	Rzyx_accel[:, :, i] = np.matmul(Rz_accel, np.matmul(Ry_accel, Rx_accel))

print('The dimensions of Rzyx_accel are: ' + str(Rzyx_accel.shape))

# Madgwick data
print('The dimensions of Magwick_x are: ' + str(Magwick_x.shape))
Rzyx_madg = np.zeros(shape = (3, 3, Magwick_x.shape[1]))
for i in range(0, Rzyx_madg.shape[2]):
	Rz_madg = np.array([[np.cos(Magwick_x[0, i]), -np.sin(Magwick_x[0, i]), 0],
		[np.sin(Magwick_x[0, i]), np.cos(Magwick_x[0, i]), 0],
		[0, 0, 1]])
	Ry_madg = np.array([[np.cos(Magwick_x[1, i]), 0, np.sin(Magwick_x[1, i])],
		[0, 1, 0],
		[-np.sin(Magwick_x[1, i]), 0, np.cos(Magwick_x[1, i])]])
	Rx_madg = np.array([[1, 0, 0],
		[0, np.cos(Magwick_x[2, i]), -np.sin(Magwick_x[2, i])],
		[0, np.sin(Magwick_x[2, i]), np.cos(Magwick_x[2, i])]])
	Rzyx_madg[:, :, i] = np.matmul(Rz_madg, np.matmul(Ry_madg, Rx_madg))

print('The dimensions of Rzyx_madg are: ' + str(Rzyx_madg.shape))



#Initialize videomaker
Framerate=15 #fps

AnimFig=plt.figure()
FFMpegWriter = manimation.writers['ffmpeg']
#if Requested MovieWriter not available error then:
#$ sudo apt install ffmpeg
metadata = dict(title='', artist='Ilya',
                comment='')
writer = FFMpegWriter(fps=Framerate, metadata=metadata)


#reduce data to frameratefps

#lets just use Vicon time as the basis

#idea from team "sudo rm -rf"
ViconTs = np.linspace(V_ts[0,0], V_ts[0,-1], num=((V_ts[0,-1]-V_ts[0,0])*Framerate) )
print('The dimensions of ViconTs are: ' + str(ViconTs.shape))
counter=0
Vicon_Rot_Ms=np.zeros((3,3,ViconTs.shape[0]))
for i in range(V_rot_m.shape[2]):
	#if you passed it grab the last one, should be close enough
	if V_ts[0,i]>ViconTs[counter]:
		Vicon_Rot_Ms[:,:,counter]=V_rot_m[:,:,i]
		counter+=1
	if counter==ViconTs.shape[0]:
		break


#IMUTs = np.linspace(ts[0,0], ts[0,-1], num=((ts[0,-1]-ts[0,0])*Framerate) )
#print('The dimensions of IMUTs are: ' + str(IMUTs.shape))

Accel_Rot_Ms=np.zeros((3,3,ViconTs.shape[0]))
Gyro_Rot_Ms=np.zeros((3,3,ViconTs.shape[0]))
Magwick_Rot_Ms=np.zeros((3,3,ViconTs.shape[0]))

#print('The dimensions of ts are: ' + str(ts.shape))

counter=0
for i in range(0,x_gyro_rad.shape[1]):

	if ts[0,i]>ViconTs[counter]:
		Accel_Rot_Ms[:,:,counter]=Rzyx_accel[:,:,i]
		Gyro_Rot_Ms[:,:,counter]=Rzyx_gyro[:,:,i]
		Magwick_Rot_Ms[:,:,counter]=Rzyx_madg[:,:,i]
		counter+=1
	if counter==ViconTs.shape[0]:
		break

#lets get plotting and writing!

#plotting setup:
Fig_animate= plt.figure(figsize=(10,4))
ax1=plt.subplot(141,projection='3d')
ax2=plt.subplot(142,projection='3d')
ax3=plt.subplot(143,projection='3d')
ax4=plt.subplot(144,projection='3d')
ax1.title.set_text('Gyro')
ax2.title.set_text('Accel')
ax3.title.set_text('Magwick')
ax4.title.set_text('Vicon')

print('writing file...')
with writer.saving(Fig_animate, filename +"_vid"+ ".mp4", ViconTs.shape[0]):
	for i in range(ViconTs.shape[0]):
		print(str(i) +'of' + str(ViconTs.shape[0]))
		ax1.clear()
		ax2.clear()
		ax3.clear()
		ax4.clear()
		ax1.title.set_text('Gyro')
		ax2.title.set_text('Accel')
		ax3.title.set_text('Magwick')
		ax4.title.set_text('Vicon')
		rotplot(Gyro_Rot_Ms[:,:,i],ax1)
		rotplot(Accel_Rot_Ms[:,:,i],ax2)
		rotplot(Magwick_Rot_Ms[:,:,i],ax3)
		rotplot(Vicon_Rot_Ms[:,:,i],ax4)
		writer.grab_frame()