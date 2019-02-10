# RoboSoccer
Teaching elementary skills to robots using motion clips and Reinforcement Learning

## Progress
	
###	Experiments
	
#### Changing learning rate 
- Too high (0.1) makes weights, means and other things explode
- Too low (0.000001) too slow in moving average change
- Medium val (0.001 -> 0.0001) works (meaning shit doesn't explode)

#### Architecture

##### Neural network ( sigma )
- Related to mu,	
- sigmoid/softplus? (ASK)
- scaled/not scaled 

##### Neural network ( mu ) 
- smaller the better (radians/sec) 2 -> 2.3 deg /0.02 sec so limit max to 3.

##### Increasing number of nodes in hidden layers
- >=100

#### Reward Funtion
- Mimic part (Pose reward (difference between join orientations))
- Velocity small (Not necessary)
- Acceration reward (difference between velocities)
- End effector reward
- Center of mass deviation penalty 

#### State Space
- 22 joint angles
- 22 actions taken
- 3 Gyroscope rate
- 3 Accelerometer rate
- 4 Postion and orientation of root
- Time variable “phase” between 0 and 1
- Cyclic motions have that phase set to 0 at end
		
#### Action Space
- 22 angular velocity values (radians/sec)


### Doubts 	
- Can we make mu span grow somehow. Coz bigger span can help only in corner cases like stopping sudden fall, but increase time of learning.
- Should we punish fallen state or/and include accelearation reward
- Getting random trajectories for the learning curve

### TODO
- Initial state distribution (Cannot place it in arbitrary position)
- Increase speed of simulation, using GPU
- Complex motions of situps, balance and handwaves
- Make it stand and balance with all 22 effectors
- Overcome Retargeting 

### Initial state distribution (TODO)
Start in all possible states in the clip rather than at the start as there is a chance that high reward states are present in the very end

### Early termination (DONE)
If body torso  or head or some other links hit the ground then terminate the learning as it is wasted in getting up from those positions
