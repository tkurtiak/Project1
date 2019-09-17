import tf
import numpy as np


# Add the Process model code to the wrappet in this format:
#import ProcessModel
#State1 = ProcessModel.ProcessModel(State0,dt)

def ProcessModel(State0,dt,w_k):
	# State is a 7x1 vector of position quaternions and angular velocity
	# State = [q0,q1,q2,q3,wx,wy,wz] 
	# dt in seconds
	# w_k is noise vector [wq,ww]

	# Initialize the next state as the current state
	State1 = np.zeros(7)

	# The simple process model A assumes constant angular velocity 
	# Over a short time period, this assumption holds well
	w_k = np.array(State0[4:6+1])
	State1[4:7] = w_k
	q_k = np.array(State0[0:4])

	# Calculate delta angle of rotation assuming constant angluar velocity over time dt
	da = dt*np.linalg.norm(w_k)
	# Calculate rotation axis for constant angular rotation
	de = w_k/np.linalg.norm(w_k)
	# Calculate change in quaternions as a result of rotation
	dq = np.array([np.cos(da/2),de[0]*np.sin(da/2),de[1]*np.sin(da/2),de[2]*np.sin(da/2)])


	# Noise Calculations
	w_q = np.array(w_k[0:3])
	a_w = np.linalg.norm(w_q)
	e_w = w_q/np.linalg.norm(w_q)
	q_w = np.array([np.cos(a_w/2),e_w[0]*np.sin(a_w/2),e_w[1]*np.sin(a_w/2),e_w[2]*np.sin(a_w/2)])
	# Assemble new state vector
	temp = tf.transformations.quaternion_multiply(q_k,q_w)
	State1[0:4] = tf.transformations.quaternion_multiply(temp,dq)
	#quaternion multiply q_k by dq to apply the rotation
	#a = q_k
	#b = q_w
	#temp = [0,0,0,0]

	#temp[0] = .5*(a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3])
	#temp[1] = .5*(a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2])
	#temp[2] = .5*(a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1])
	#temp[3] = .5*(a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0])

	#a = temp
	#b = dq
	#temp2[0] = .5*(a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3])
	#temp2[1] = .5*(a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2])
	#temp2[2] = .5*(a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1])
	#temp2[3] = .5*(a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0])

	#State1[0:4] = temp2
	#State1[4:7] = State1[4:7]+w_k[4:7]
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
	
	temp = tf.transformations.quaternion_multiply(q_k,g)
	qi = np.array([q_k[0],-q_k[1],-q_k[2],-q_k[3]])
	gprime = tf.transformations.quaternion_multiply(temp,qi) 
	# Reccomend use tf for quaternion multiply

	Accel1 = gprime+v[0:4]

	return Accel1


	# For Testing
	# import ProcessModel as model 
	# State0 = [0,0,0,.2,0,0,1]	 
	# model.MeasurementModel_Accel(State0,[0,0,0,0,0,0,0])
