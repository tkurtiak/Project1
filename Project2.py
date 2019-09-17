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
filename = "imuRaw10"
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
	while error_val>0.001:

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
			error_vector= error_matrix[1:,i]/(np.sin(theta/2))
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

def COV_statevectors(state_vectors,xbar=np.array([0.,0.,0.,0.,0.,0.,0.])):
	#so you got a bunch of 7d state vectors, and their average
	#give back a 7x7 covariance matrix, itll always be 7x7 for this input

	try:
		n=float(state_vectors.shape[1])
	except:
		n=1.
		

	temp=np.zeros((7,1))
	cov_matrix=np.zeros((7,7))
	for i in range(int(n)):

		if(n==1):
			temp[:,0]=state_vectors-xbar
		else:	
			temp[:,0]=state_vectors[:,i]-xbar
		#print(temp.shape)
		temp_matrix=np.matmul(temp,np.transpose(temp))
		#print(temp_matrix.shape)
		cov_matrix= cov_matrix+(temp_matrix/n)
		#print(cov_matrix)
	return cov_matrix

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



# Measurement Model for Gyroscope
def MeasurementModel_Gyro(State0,v):
	# State is a 7x1 vector of position quaternions and angular velocity
	# State = [q0,q1,q2,q3,wx,wy,wz] 
	# v is 7x1 noise vector
	Gyro1 = np.zeros(3)
	Gyro1 = np.array(State0[4:7])+np.array(v[4:7])

	return Gyro1



## NOT FINISHED
	# Measurement Model for Accelerometer
def MeasurementModel_Accel(State0,v):
	# State is a 7x1 vector of position quaternions and angular velocity
	# State = [q0,q1,q2,q3,wx,wy,wz] 
	# v is 7x1 noise vector
	Accel1 = np.zeros(4)
	q_k = np.array(State0[0:4])
	# Define g as gravity down, ie a_z = -9.81 m/s/s
	g = np.array([0,0,0,-9.81])
	
	temp = quat_mult(q_k,g)
	qi = np.array([q_k[0],-q_k[1],-q_k[2],-q_k[3]])
	gprime = quat_mult(temp,qi) 
	# Reccomend use tf for quaternion multiply

	Accel1 = gprime+v[0:4]

	return Accel1




#MAIN_------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
#TEMP DATA
n=7.
#state_vectorZ=np.random.rand(7,int(n)) #each column is a state vector x=[q w]'
#for i in range(int(n)):
#	state_vectorZ[0:4,i]=state_vectorZ[0:4,i]*(1/np.linalg.norm(state_vectorZ[0:4,i]))

#letf find the process noise vector real quick


#computation of the mean of state vector x
#x_bar=AVG_statevectors(state_vectorZ)
#print(x_bar)

#cov_m=COV_statevectors(state_vectorZ,x_bar)
#print(cov_m)

#print('trying 1d covariance')
#print(state_vectorZ[:,1])
#cov_m=COV_statevectors(state_vectorZ[:,1])
#print(cov_m)


#initialize some stuff???
#6D noise vector,#letf find the process noise vector real quick
w_k=np.array([np.cov(x_gyro_rad[0,0:200]),np.cov(x_gyro_rad[1,0:200]),np.cov(x_gyro_rad[2,0:200]),np.cov(w_rad[0,0:200]),np.cov(w_rad[1,0:200]),np.cov(w_rad[2,0:200])])

# Noise Calculations
w_q = np.array(w_k[0:3])
a_w = np.linalg.norm(w_q)
e_w = w_q/np.linalg.norm(w_q)
q_w = np.array([np.cos(a_w/2),e_w[0]*np.sin(a_w/2),e_w[1]*np.sin(a_w/2),e_w[2]*np.sin(a_w/2)])


#lets take an initial sample of states, first 200, also lets just assume x_mean to avoid a heavy mean calc
initial_states=np.zeros((7,200))
for i in range(200):
	phi=x_gyro_rad[0,i]
	theta=x_gyro_rad[1,i]
	psi=x_gyro_rad[2,i]
	QX=np.array([np.cos(phi/2),np.sin(phi/2),0.,0.])
	QY=np.array([np.cos(theta/2),0.,np.sin(theta/2),0.])
	QZ=np.array([np.cos(psi/2),0.,0.,np.sin(psi/2)])
	initial_states[0:4,i]= quat_mult(QZ,quat_mult(QY,QX))
	initial_states[4:,i]=w_rad[:,i]


#previous state vector x_k-1 initialize to 0's baseline:
x_prev=np.array([1.,0.,0.,0.,0.,0.,0.])
#its covariance P_k-1

P_prev=COV_statevectors(initial_states,x_prev)



#Cholesky square root
S=np.transpose(np.linalg.cholesky(P_prev))
#print('confirming')
#print(P_prev)
#print('P_prev ^ StS down')
#print(np.matmul(np.transpose(S),S))

W= np.concatenate((S*(np.sqrt(2*n)),S*(-1*np.sqrt(2*n))),axis=1)

X=np.zeros(W.shape) #option exists here to add the mean in as well with a weight


for i in range(W.shape[1]):
	X[:,i]= x_prev + W[:,i]

#gonna go ahead and skip the process noise covariance addition because like.. wtf even is it?

Y=np.zeros(X.shape)
#print(ts.shape)
dt=ts[0,200]-ts[0,199]
for i in range(X.shape[1]):
	print(np.linalg.norm(X[:4,i]))
	Y[:,i]= ProcessModel(X[:,i],dt)

#print(Y.shape)
#print(Y)

x_prior=AVG_statevectors(Y)
P_k_prior=COV_statevectors(Y,x_prior)
#print(x_bar)

#cov_m=COV_statevectors(state_vectorZ,x_bar)
#print(cov_m)
