from scipy import io
import matplotlib.pyplot as plt
import numpy as np 


def quat_mult(a,b):
	
	c = np.array([0.,0.,0.,0.])
	c[0] = (a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3] )
	c[1] = (a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2] )
	c[2] = (a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1] )
	c[3] = (a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0] )
	return c



#Change file name here to view various data sets
filename = "imuRaw6"
filename2 = "viconRot6"

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




def AVG_statevectors(state_vectors,mean_quat=np.array([1.,0.,0.,0.])):
	#make state vectors 7 rows by n column vectors, every column is 7d [quat, p,q,r] state vector
	n=float(state_vectors.shape[1])
	#normalize the quaternions (just in case)
	for i in range(int(n)):
		state_vectors[0:4,i]=state_vectors[0:4,i]*(1/np.linalg.norm(state_vectors[0:4,i]))

	
	mean_inv=np.array([1.,0.,0.,0.])
	error_matrix=np.zeros((4,int(n)))
	error_val=100.0
	#in case of infinite loop debug with counter
	counter=0
	while error_val>0.01:

		#this for debugging infinite loop
		counter=counter+1
		#print('counter is ' + str(counter) + ' error val ' + str(error_val))

		if(counter>10000):
			print('counter is ' + str(counter) + ' error val ' + str(error_val) + '.   BREAKING')
			break

		#inverse the mean
		mean_inv[0]=mean_quat[0]
		mean_inv[1:]=-1*mean_quat[1:]

		#initialize error_vector
		error_vector_mean=np.array([0.,0.,0.])

		for i in range(int(n)):
			#calc error quaternions

			error_matrix[:,i]= quat_mult(state_vectors[0:4,i],mean_inv)

			#normalize (just in case)
			error_matrix[:,i]= error_matrix[:,i]*(1/np.linalg.norm(error_matrix[:,i]))

			#you have quaternion, now make into error vector subscript i
			theta= 2*np.arccos(error_matrix[0,i])

			if (theta!=0):
				error_vector= error_matrix[1:,i]/(np.sin(theta/2))
			else:
				error_vector= error_matrix[1:,i]
				
			#add it to the mean
			error_vector_mean= error_vector_mean + (error_vector/n)

		#heres your convergence value, when error vector is 0,0,0 you're good, this should be close enough
		error_val=np.linalg.norm(error_vector_mean)

		#okay now convert your mean error vector back into a quaternion to update the mean quaternion
		error_quat=np.array([np.cos(error_val/2), error_vector_mean[0]*np.sin(error_val/2),error_vector_mean[1]*np.sin(error_val/2),error_vector_mean[2]*np.sin(error_val/2)])
		#normalize for those floating point errors and stuff
		error_quat= error_quat*(1/np.linalg.norm(error_quat))

		#update your mean quaternion
		mean_quat=quat_mult(error_quat,mean_quat)
		#normalize that too
		mean_quat=mean_quat*(1/np.linalg.norm(mean_quat))
	#holy hell, finally have mean quat
	#alright finish up the mean with regular averaging

	#initialize the mean rates and average them up
	mean_rates=np.array([0.,0.,0.])	
	for i in range(int(n)):
		mean_rates=mean_rates + state_vectors[4:,i]*(1/n)
	#put it all together
	xbar=np.concatenate((mean_quat, mean_rates))

	return xbar

def COV_6d(state_vectors,xbar=np.array([0.,0.,0.,0.,0.,0.])):
	#so you got a bunch of 6d state vectors, and their average
	#give back a 6x6 covariance matrix, itll always be 6x6 for this input
	oned=0
	try:
		n=float(state_vectors.shape[1])
	except:
		n=1.
		oned=1
		

	temp=np.zeros((6,1))
	cov_matrix=np.zeros((6,6))
	for i in range(int(n)):

		if(oned==1):
			temp[:,0]=state_vectors-xbar
		else:	
			temp[:,0]=state_vectors[:,i]-xbar
		#print(temp.shape)
		temp_matrix=np.matmul(temp,np.transpose(temp))
		#print(temp_matrix.shape)
		cov_matrix= cov_matrix+(temp_matrix/n)
		#print(cov_matrix)
	return cov_matrix

def COV_7d(state_vectors,xbar=np.array([1.,0.,0.,0.,0.,0.,0.])):
	#alright so we got these 7d vectors right, so we want to get some sick ass 6d covariance matrix out of it
	try:
		n=float(state_vectors.shape[1])
	except:
		n=1.	
	temp=np.zeros((6,1))
	cov_matrix=np.zeros((6,6))
	Wprime=np.zeros((6,int(n)))
	
	for i in range(int(n)):
		#so we need to grab this quaternion shit and find the error quaternions (subtract out the mean quaternion)
		if(n==1):
			e_quat=quat_mult(state_vectors[:4],np.array([xbar[0],-xbar[1],-xbar[2],-xbar[3]]))
		else:	
			e_quat=quat_mult(state_vectors[:4,i],np.array([xbar[0],-xbar[1],-xbar[2],-xbar[3]]))
		#okay lets put this bitch ass error quaternion into a rotation vector thingy thang
		theta= 2*np.arccos(e_quat[0]) #angle
		if (np.abs(np.abs(e_quat[0])-1)>0.01):
			axis= e_quat[1:]/(np.sin(theta/2)) #vector
		else:
			axis= e_quat[1:] #if angle is 0 then this bitch boi is also all 0s
		#okay bet so now we grab this a_w and e_w things from like equation 14,15 and make into w_q, which is r_w now for this thing because consistent notation is for CHUMPS
		r_w=theta*axis
		#alright now gg2ez subtract out the rates
		if(n==1):
			w_w=state_vectors[4:]-xbar[4:]
		else:	
			w_w=state_vectors[4:,i]-xbar[4:]
		#bet now we can get these W'i vectors that are basically X-xbar so we can do the transposy thingy and mean it all out
		Wprime[:,i]=np.array([r_w[0],r_w[1],r_w[2],w_w[0],w_w[1],w_w[2]]) #could have concat but didnt want to deal with axis and dimensions and shit
		temp[:,0]=Wprime[:,i]
		temp_matrix=np.matmul(temp,np.transpose(temp))
		cov_matrix= cov_matrix+(temp_matrix/n)

	#bet
	return cov_matrix,Wprime
				


def ProcessModel(State0,dt):
	# State is a 7x1 vector of position quaternions and angular velocity
	# State = [q0,q1,q2,q3,wx,wy,wz] 
	# dt in seconds
	# w_k is noise vector [wq,ww] NOT USED

	# Initialize the next state as the current state
	State1 = np.zeros(7)

	# The simple process model A assumes constant angular velocity 
	# Over a short time period, this assumption holds well
	w_k = np.array(State0[4:])
	State1[4:7] = w_k
	q_k = np.array(State0[:4])

	# Calculate delta angle of rotation assuming constant angluar velocity over time dt
	mag= np.linalg.norm(w_k)
	da = dt*mag
	# Calculate rotation axis for constant angular rotation

	if mag != 0:
		de = w_k/np.linalg.norm(w_k)
	else:
		de=w_k
	# Calculate change in quaternions as a result of rotation
	dq = np.array([np.cos(da/2),de[0]*np.sin(da/2),de[1]*np.sin(da/2),de[2]*np.sin(da/2)])


	# Noise Calculations
	#w_q = np.array(w_k[0:3])
	#a_w = np.linalg.norm(w_q)
	#e_w = w_q/np.linalg.norm(w_q)
	#q_w = np.array([np.cos(a_w/2),e_w[0]*np.sin(a_w/2),e_w[1]*np.sin(a_w/2),e_w[2]*np.sin(a_w/2)])
	# Assemble new state vector
	#temp = quat_mult(q_k,q_w)
	State1[0:4] = quat_mult(q_k,dq)
	
	return State1



# Measurement Model 
def MeasurementModel(State0):
	# State is a 7x1 vector of position quaternions and angular velocity
	# State = [q0,q1,q2,q3,wx,wy,wz] 
	# v is 7x1 noise vector NOT USED
	Gyro1 = np.zeros(3)
	Gyro1 = np.array(State0[4:7])#+np.array(v[4:7])

	Accel1 = np.zeros(4)
	q_k = np.array(State0[0:4])
	# Define g as gravity down, ie a_z = -9.81 m/s/s
	g = np.array([0,0,0,9.81])
	
	# inverse q_k is the negative of the last 3 components
	qi = np.array([q_k[0],-q_k[1],-q_k[2],-q_k[3]])

	temp = quat_mult(qi,g)
	
	gprime = quat_mult(temp,q_k) 
	

	Accel1 = gprime[1:4]#+v[0:4]
	#put it all together
	Z = np.concatenate((Accel1,Gyro1),axis=None)
	return Z






#MAIN_------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------



#TUNING PARAM
R=np.zeros((6,6)) #Measurement uncertainty covariance: idk what else to set this to rn
Ra = 0.01
Rg = 0.1
R = np.diag(np.concatenate([Ra*np.ones([1,3]),Rg*np.ones([1,3])],axis=None))

Q=np.zeros((6,6)) #Measurement uncertainty covariance: idk what else to set this to rn
Qa = .01
Qg = 0.1
Q = np.diag(np.concatenate([Qa*np.ones([1,3]),Qg*np.ones([1,3])],axis=None))


#Here are some states from just pure gyro data and their covariance
#init_mean=np.array([np.mean(x_gyro_rad[0,:200]),np.mean(x_gyro_rad[1,:200]),np.mean(x_gyro_rad[2,:200]),np.mean(w_rad[0,:200]),np.mean(w_rad[1,:200]),np.mean(w_rad[2,:200])])
#init_states=np.zeros((6,200))
#for i in range(200):
#	init_states[:,i]=np.concatenate((x_gyro_rad[:,i],w_rad[:,i]),axis=None)
##Q=np.matmul(w_k,np.transpose(w_k))
#Q=COV_6d(init_states,init_mean)



#INITIALIZE
#previous state vector x_k-1 initialize to 0's baseline:
x_prev=np.array([1.,0.,0.,0.,0.,0.,0.])
#its covariance P_k-1
#lets initialize at 0, this gets updated as we go so fuck it as long as its mildly valid
P_prev=np.ones((6,6))
#P_prev=np.diag(0.1*np.ones(6))


Xs_filter=np.zeros((7,ts.shape[1]))
UKF_out=np.zeros((6,ts.shape[1]))
Xs_filter[:,0]=x_prev



#ALLLLLLLRIGHTTYYYY THEN WE START THE LOOPAROONI NOW

for I_time in range(ts.shape[1]-1): #fuck you ill do this later (uncomment this) #################################################################################
	#I_time=1 #for now you asshole
	print(I_time)

	#Cholesky square root


	S=np.transpose(np.linalg.cholesky(P_prev+Q))
	#print('confirming')
	#print(P_prev)
	#print('P_prev ^ StS down')
	#print(np.matmul(np.transpose(S),S))

	#take the columns of S to eat up W
	W= np.concatenate((S*(np.sqrt(2*6)),S*(-1*np.sqrt(2*6))),axis=1) 
	#print('W:')
	#print(W)

	#Make the sigma points
	X=np.zeros((7,W.shape[1])) #alrighhtttty so X needs to be a 7 d boiii
	#lets throw x_prev into the W boi witha quaternion conversion thingy
	for i in range(W.shape[1]):
		temp_rot_v=W[:3,i] #first three are the rotation angles, second 3 are the rates so...we grab the angles to make a quaternion transform
		a_w=np.linalg.norm(W[:4,i])#angle
		e_w=W[:3,i]/a_w#rot vecotor
		q_w=np.array([np.cos(a_w/2),e_w[0]*np.sin(a_w/2),e_w[1]*np.sin(a_w/2),e_w[2]*np.sin(a_w/2)])#convert to quaternion
		X[:4,i]= quat_mult(x_prev[:4],W[:4,i]/np.linalg.norm(W[:4,i]))#shove it in there
		X[:4,i]=X[:4,i]/np.linalg.norm(X[:4,i])#just in fucking case yo
		X[4:,i]= x_prev[4:] + W[3:,i]#gg2ez with the rotation rates


	#Alrighty, lets tranform these sigma points BOIIII
	Y=np.zeros(X.shape)
	dt=ts[0,I_time+1]-ts[0,I_time]
	for i in range(X.shape[1]):
		Y[:,i]= ProcessModel(X[:,i],dt)

	#Now with Y we can pull out x_kminus, Wiprime,and Pkminus which is step 4,5,6
	x_prior=AVG_statevectors(Y)
	P_k_prior,Wprime=COV_7d(Y,x_prior)

	



	# Step 7, Calculate Expected Sensor output Z at predicted states Y
	Z=np.zeros((6,Y.shape[1]))
	for i in range(Y.shape[1]):
		Z[:,i] = MeasurementModel(Y[:,i])
	z_bar=np.mean(Z,axis=1)


	#Step 8
	SensorState = np.concatenate((a_scaled[:,I_time],x_gyro_rad[:,I_time]),axis=None)
	innovation=np.zeros((6,1))
	innovation[:,0] = SensorState - z_bar

	#Step 9
	P_zz=COV_6d(Z,z_bar)

	Pvv=P_zz+R

	#Cross Correlation Matrix
	Pxz= np.zeros((6,6))
	for i in range(12):
		temp_matrix= np.outer(Wprime[:,i],(Z[:,i]-z_bar))
		Pxz=Pxz+(temp_matrix/12)

	Kk= np.matmul(Pxz,np.linalg.inv(Pvv))

	#okay so we cant just add a 6d to a 7d, IE( x_posteriori=x_prior+ np.matmul(Kk,innovation) )
	x_posteriori=np.zeros(7)

	Kalman6d=np.matmul(Kk,innovation)
	#print(Kalman6d.shape)
	# so... gonna have to take the 6d and make it 7
	#so build the quaternion
	mag= np.linalg.norm(Kalman6d[:3,0])
	if mag != 0:
		de = Kalman6d[:3,0]/mag
	else:
		de=Kalman6d[:3,0]
	KalmanQ = np.array([np.cos(mag/2),de[0]*np.sin(mag/2),de[1]*np.sin(mag/2),de[2]*np.sin(mag/2)])

	x_posteriori[:4]=quat_mult(x_prior[:4],KalmanQ)
	#okay now do the state vector addition thingy
	x_posteriori[4:]=Kalman6d[3:,0]+x_prior[4:]

	#update P

	


	P_prev=P_k_prior - np.matmul(np.matmul(Kk,Pvv),np.transpose(Kk))
	#reset x for next round
	x_prior=x_posteriori

	Xs_filter[:,I_time+1]=x_posteriori

	#convert to Euler angles for plots
	#phi
	UKF_out[2,I_time+1]= np.arctan2((2*(x_posteriori[0]*x_posteriori[1] + x_posteriori[2]*x_posteriori[3])),(1-2*(x_posteriori[1]**2 + x_posteriori[2]**2)))
	#theta
	UKF_out[1,I_time+1]= np.arcsin(2*(x_posteriori[0]*x_posteriori[2] - x_posteriori[3]*x_posteriori[1]))
	#psi
	UKF_out[0,I_time+1]= np.arctan2((2*(x_posteriori[0]*x_posteriori[3] + x_posteriori[1]*x_posteriori[2])),(1-2*(x_posteriori[2]**2 + x_posteriori[3]**2)))
	UKF_out[3:,I_time+1] = x_posteriori[4:]

	#print('P_k_prior')
	#print(P_k_prior)
	print('Y')
	print(Y)
	print('x_prior')
	print(x_prior)
	#print('x_prior quat mag')
	#print(np.linalg.norm(x_prior[:4]))

	print('Wprime')
	print(Wprime)
	#print('Wprime[:,11],np.transpose(Wprime[:,11]')
	#print(np.matmul(Wprime[:,11],np.transpose(Wprime[:,11])))
	#print('Wprime[:,11].shape')
	#print(Wprime[:,11].shape)






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
ax1.plot(plot_time,UKF_out[0,:],label = 'UKF')

plt.ylabel('angle, Radians')
plt.legend(loc = 'lower right')

ax2 = fig.add_subplot(3,1,2,title = 'Pitch, Theta')
ax2.plot(plot_time_V,V_x_rad[1,:],label = 'Vicon')
plt.hold(True)
ax2.plot(plot_time,x_gyro_rad[1,:],label = 'Gyro')
ax2.plot(plot_time,a_orientation[1,:],label = 'Accel')
ax2.plot(plot_time,UKF_out[1,:],label = 'UKF')

plt.ylabel('angle, Radians')
plt.legend(loc = 'lower right')

ax3 = fig.add_subplot(3,1,3,title = 'Roll, Phi')
ax3.plot(plot_time_V,V_x_rad[2,:],label = 'Vicon')
plt.hold(True)
ax3.plot(plot_time,x_gyro_rad[0,:],label = 'Gyro')
ax3.plot(plot_time,a_orientation[0,:],label = 'Accel')
ax3.plot(plot_time,UKF_out[2,:],label = 'UKF')
plt.xlabel('time, s')
plt.ylabel('angle, Radians')
plt.legend(loc = 'lower right')

plt.savefig(filename + ".png")
plt.show()