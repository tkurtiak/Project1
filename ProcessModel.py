#import tf
import numpy as np


# Add the Process model code to the wrappet in this format:
#import ProcessModel
#State1 = ProcessModel.ProcessModel(State0,dt)

def ProcessModel(State0,dt):
	# State is a 7x1 vector of position quaternions and angular velocity
	# State = [q0,q1,q2,q3,wx,wy,wz] 

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

	# Assemble new state vector
	# Had an issue with tf due to sorcing...
	#State1[0:4] = tf.transformations.quaternion_multiply(q_k,dq)
	#quaternion multiply q_k by dq to apply the rotation
	a = q_k
	b = dq
	temp = [0,0,0,0]
	temp[0] = .5*(a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3])
	temp[1] = .5*(a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2] )
	temp[2] = .5*(a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1] )
	temp[3] = .5*(a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0] )
	State1[0:4] = temp
	return State1