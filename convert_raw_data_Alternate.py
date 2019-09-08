from scipy import io
import matplotlib.pyplot as plt
import numpy as np 
#from pyquaternion import Quaternion
import tf
#from tf.transformations import quaternion_from_euler
#from scipy.spatial.transform import Rotation
#pip install pyquaternion

filename = "imuRaw1"
filename2 = "viconRot1"
file = io.loadmat("Data/Train/IMU/"+filename+".mat")
VICONfile = io.loadmat("Data/Train/Vicon/"+filename2+".mat")
IMUparams = io.loadmat("IMUParams.mat")

raw = file['vals']
ts = file['ts']
#print(raw.shape)
IMUparams_raw = IMUparams['IMUParams']
#How to retrive data:   raw[which dtyp (ax,ay,az,wz,wx,wy), index of var]
#print(IMUparams_raw.shape)
#print(ts.shape) 

w_raw = np.array([raw[4,:],raw[5,:],raw[3,:]]) #reorder for x, y, z
a_raw = np.array([raw[0,:],raw[1,:],raw[2,:]]) 
#print(w_raw.shape)

# Calculate Bias as the average of the first 300 gyro datapoints when the gyro is at rest (assumed)
bias_w = np.array([w_raw[0,0:200].mean(),w_raw[1,0:200].mean(),w_raw[2,0:200].mean()])
# extract accelerometer bias from IMU params.  b_ax, b_ay, b_az
bias_a = np.array([IMUparams_raw[1,0],IMUparams_raw[1,1],IMUparams_raw[1,2]])
# extract accelerometer bias from IMU params.  s_x, s_y, s_z
scale_a = np.array([IMUparams_raw[0,0],IMUparams_raw[0,1],IMUparams_raw[0,2]])

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

# Calculate Accelerometer data [ax,ay,az]
#a_x = (a_raw[0,:]+bias_a[0])/scale_a[0]
#a_y = (a_raw[1,:]+bias_a[1])/scale_a[1]
#a_z = (a_raw[2,:]+bias_a[2])/scale_a[2]

# Basic data doesnt make sense.  Try multiplying by scale instead of dividing
#a_x2 = (a_raw[0,:]+bias_a[0])*scale_a[0]
#a_y2 = (a_raw[1,:]+bias_a[1])*scale_a[1]
#a_z2 = (a_raw[2,:]+bias_a[2])*scale_a[2]

# Basic data doesnt make sense.  Try multiplying by bias and adding scale instead
#a_x3 = (a_raw[0,:]+scale_a[0])/bias_a[0]
#a_y3 = (a_raw[1,:]+scale_a[1])/bias_a[1]
#a_z3 = (a_raw[2,:]+scale_a[2])/bias_a[2]

# Basic data doesnt make sense.  Try scaling first
a_x4 = (a_raw[0,:]*scale_a[0]+bias_a[0])*9.81
a_y4 = (a_raw[1,:]*scale_a[1]+bias_a[1])*9.81
a_z4 = (a_raw[2,:]*scale_a[2]+bias_a[2])*9.81



a_scaled = np.array([a_x4,a_y4,a_z4])

#print("a_scaled shape")
#print(a_scaled.shape)
#state vector is x: phi theta psi Euler angles (roll,pitch,yaw)
x_gyro_rad=np.zeros(w_rad.shape)
a_orientation = np.zeros(a_scaled.shape)

for i in range(x_gyro_rad.shape[1]):
	
	#integrates the gyro data
	temp=np.trapz(w_rad[:,:i],axis=1,x=ts[0,:i])
	
	x_gyro_rad[:,i] = np.transpose(temp)


#	a_phi = np.arctan(a_y4[i]/np.sqrt(a_x4[i]*a_x4[i]+a_z[i]*a_z[i]))
#	a_theta = np.arctan(a_x4[i]/np.sqrt(a_y4[i]**2+a_z4[i]**2))
#	a_psi = np.arctan(np.sqrt(a_x4[i]**2+a_y4[i]**2)/a_z4[i])

	#COMPUTES orientation from accelerometer data
	a_phi2 = np.arctan2(a_y4[i],np.sqrt((a_x4[i]**2)+(a_z4[i]**2)))
	a_theta2 = np.arctan2(-a_x4[i],np.sqrt((a_y4[i]**2)+(a_z4[i]**2)))
	a_psi2 = np.arctan2(np.sqrt((a_x4[i]**2)+(a_y4[i]**2)),a_z4[i])


	a_orientation_temp = np.array([a_phi2,a_theta2,a_psi2])
	a_orientation[:,i] = a_orientation_temp
#if i==0:
#	 a_zero = a_orientation[:,0]
#a_orientation[:,i] = a_orientation_temp - a_zero


#Adjust to keep in +-pi range (DONT NEED?)
# for i in range(x_gyro_rad.shape[1]):
# 	for j in range(x_gyro_rad.shape[0]):
# 		if x_gyro_rad[j,i]> np.pi:
# 			x_gyro_rad[j,i] = x_gyro_rad[1,i]#-2*np.pi
			
# 		if x_gyro_rad[j,i]< -np.pi:
# 			x_gyro_rad[j,i] = x_gyro_rad[1,i]#+2*np.pi





# Now Process VICON data
V_ts= VICONfile['ts']
V_rot_m= VICONfile['rots']


V_x_rad=np.zeros((3,V_rot_m.shape[2]))

#print(V_rot_m.shape)


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


# Check this, Z-Y-X Euler Angle Assignments

	V_x_rad[0,i]=phi1
	V_x_rad[1,i]=th1
	V_x_rad[2,i]=psi1



print('done')




#ON TO THE FILTER
print('starting filter')
######################################################################
#####TUNE THIS############
Beta=0.1
#####################################################################

Magwick_x= np.zeros([3,x_gyro_rad.shape[1]])
est_quat= np.zeros([4,x_gyro_rad.shape[1]])
est_quat[0,0]=1 #let initial rotation be null rotation

qdot_est= np.zeros([4,x_gyro_rad.shape[1]]) #idk about null initial rotation speed, normal velocity vector not a requisite
qdot_est[0,0]=0


#print(np.linalg.norm(np.array([2,2,0])))

#cant go all the way to end since putting in state at t+1 every time
for i in range(x_gyro_rad.shape[1]-1):
	#start with something
	q_est_t=est_quat[:,i]
	

	#GYRO BRANCH
	#build the right side from slide 17 lecture 2b
	#tempquat = tf.transformations.quaternion_from_euler(x_gyro_rad[0,i+1],x_gyro_rad[1,i+1],x_gyro_rad[2,i+1],'rzyx')
	
	#tempquat = [ 0, x_gyro_rad[0,i+1],x_gyro_rad[1,i+1],x_gyro_rad[2,i+1]]
	tempquat = [ 0, w_rad[0,i+1],w_rad[1,i+1],w_rad[2,i+1]]
	
	#tempquat=Quaternion(tempquat_0)
	#make q_est_t into actual quaternion for multiply
	#temp= Quaternion(q_est_t)
	#temp = np.array[q_est_t[0],q_est_t[1],q_est_t[2],q_est_t[3]]
	temp = q_est_t
	#orien_incr_quat= tf.transformations.quaternion_multiply(.5*temp,tempquat)
	a = temp
	b = tempquat
	orien_incr_quat = [0,0,0,0]
	orien_incr_quat[0] = .5*(a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3])
	orien_incr_quat[1] = .5*(a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2] )
	orien_incr_quat[2] = .5*(a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1] )
	orien_incr_quat[3] = .5*(a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0] )
	#retrieve quaternion and convert to numpy array
	#orientation_inc= np.transpose(np.array([orien_incr_quat.real, orien_incr_quat.vector[0], orien_incr_quat.vector[1], orien_incr_quat.vector[2]]))
	
	#orientation_inc= np.transpose(np.array([orien_incr_quat[0],orien_incr_quat[1],orien_incr_quat[2],orien_incr_quat[3]]))
	orientation_inc= np.array([orien_incr_quat[0],orien_incr_quat[1],orien_incr_quat[2],orien_incr_quat[3]])

	#ACCEL BRANCH
	#equation is f(q_est,t, g, a_t+1) so
	#looking at the actual paper seems like equation on slide 20 is wrong
	#del f should be function of q_t not q_t+1 

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
	qdot_est[:,i+1]=orientation_inc+grad_func

	#INTEGRATE
	ii=i+1
	#temp=np.trapz(qdot_est[:,:ii],axis=1,x=ts[0,:ii])
	dt= ts[0,ii]-ts[0,i]
	temp2= q_est_t + dt*qdot_est[:,i+1]
	#print('integrating:')
	#print(qdot_est[:,:ii])
	#print(ts[0,:ii])
	#print(temp)
	#normalize before saving
	if np.count_nonzero(temp2)==0:
		temp2[0]=1
		#print('got into if')
	temp2=temp2/(np.linalg.norm(temp2))
	
	est_quat[:,i+1] = np.transpose(temp2)
	#est_quat[:,i+1] = temp2

	#r_rotation=Rotation.from_quat(temp)
	EulerOut = tf.transformations.euler_from_quaternion(temp2,'szyx')

	#r_rotation.as_euler('zyx')
	#Magwick_x[0,i+1]=r_rotation.as_euler('zyx')[2] #phi
	#Magwick_x[1,i+1]=r_rotation.as_euler('zyx')[1] #theta
	#Magwick_x[2,i+1]=r_rotation.as_euler('zyx')[0] #psi
	
	#Magwick_x[2,i+1]=EulerOut[2] #phi
	Magwick_x[2,i+1]= -np.arctan2(2*temp2[1]*temp2[2]-2*temp2[0]*temp2[3],2*temp2[0]**2+2*temp2[1]**2-1) 
	
	Magwick_x[1,i+1]= EulerOut[1] #theta
#	Magwick_x[0,i+1]=-EulerOut[0] #psi
	Magwick_x[0,i+1]=-np.arctan2(2*temp2[2]*temp2[3]-2*temp2[0]*temp2[1],2*temp2[0]**2+2*temp2[3]**2-1)





print('plotting')
#print(np.transpose(V_ts[:]).shape)
#print(V_x_rad[0,:].shape)
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
#ax1.title('Psi')
ax1.plot(plot_time_V,V_x_rad[0,:],label = 'Vicon')
plt.hold(True)
ax1.plot(plot_time,x_gyro_rad[2,:],label = 'Gyro')
ax1.plot(plot_time,a_orientation[2,:],label = 'Accel')
ax1.plot(plot_time,Magwick_x[2,:],label = 'Magwick')
#plt.xlabel('time, s')
plt.ylabel('angle, Radians')
plt.legend(loc = 'lower right')

ax2 = fig.add_subplot(3,1,2,title = 'Pitch, Theta')
ax2.plot(plot_time_V,V_x_rad[1,:],label = 'Vicon')
plt.hold(True)
ax2.plot(plot_time,x_gyro_rad[1,:],label = 'Gyro')
ax2.plot(plot_time,a_orientation[1,:],label = 'Accel')
ax2.plot(plot_time,Magwick_x[1,:],label = 'Magwick')
#plt.xlabel('time, s')
plt.ylabel('angle, Radians')
plt.legend(loc = 'lower right')

ax3 = fig.add_subplot(3,1,3,title = 'Roll, Phi')
ax3.plot(plot_time_V,V_x_rad[2,:],label = 'Vicon')
plt.hold(True)
ax3.plot(plot_time,x_gyro_rad[0,:],label = 'Gyro')
ax3.plot(plot_time,a_orientation[0,:],label = 'Accel')
ax3.plot(plot_time,Magwick_x[0,:],label = 'Magwick')
plt.xlabel('time, s')
plt.ylabel('angle, Radians')
plt.legend(loc = 'lower right')

#mng = plt.get_current_fig_manager()
#mng.window.state('zoomed')

plt.savefig(filename + ".png")
plt.show()
#fig2 = plt.figure()
#ax1 = fig2.add_subplot(4,1,1,title = 'q0')
#ax1.plot(np.transpose(ts),est_quat[0,:],label = 'ViconPsi')

#ax2 = fig2.add_subplot(4,1,2,title = 'q1')
#ax2.plot(np.transpose(ts),est_quat[1,:],label = 'ViconPsi')

#ax3 = fig2.add_subplot(4,1,3,title = 'q2')
#ax3.plot(np.transpose(ts),est_quat[2,:],label = 'ViconPsi')

#ax4 = fig2.add_subplot(4,1,4,title = 'q3')
#ax4.plot(np.transpose(ts),est_quat[3,:],label = 'ViconPsi')
#plt.show()

#fig3 = plt.figure()
#ax1 = fig3.add_subplot(4,1,1,title = 'qdot0')
#ax1.plot(np.transpose(ts),qdot_est[0,:],label = 'ViconPsi')

#ax2 = fig3.add_subplot(4,1,2,title = 'qdot1')
#ax2.plot(np.transpose(ts),qdot_est[1,:],label = 'ViconPsi')

#ax3 = fig3.add_subplot(4,1,3,title = 'qdot2')
#ax3.plot(np.transpose(ts),qdot_est[2,:],label = 'ViconPsi')

#ax4 = fig3.add_subplot(4,1,4,title = 'qdot3')
#ax4.plot(np.transpose(ts),qdot_est[3,:],label = 'ViconPsi')
#plt.show()


#raw_input('Press Enter to exit')