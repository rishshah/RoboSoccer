# RoboSoccer
Teaching elementary skills to robots using motion clips and Reinforcement Learning

## Overview

### States
- Relative position of all links, their orientation in quaternions, linear and angular velocities.
- Time variable “phase” between 0 and 1
- Cyclic motions have that phase set to 0 at end

### Actions
- Target orientation at each joints


### Reward
- Imitation reward
- Pose reward (difference between join orientations)
- Velocity reward (difference between velocities)
- End effector reward
- Center of mass deviation penalty 
- Acceleration reward

### Initial state distribution (TODO)
Start in all possible states in the clip rather than at the start as there is a chance that high reward states are present in the very end

### Early termination (DONE)
If body torso  or head or some other links hit the ground then terminate the learning as it is wasted in getting up from those positions

## Progress
	
###	Experiments
	
#### Changing learning rate 
- Too high (0.1) makes weights, means and other things explode
- Too low (0.000001) too slow in moving average change
- Medium val (0.001 -> 0.0001) works (meaning shit doesn't explode)

#### Architecture

##### Neural network ( sigma ) (not much difference in outcome)
- not related to mu,	
- realted to mu,
- sigmoid/softplus ? (ASK)
- scaled/not scaled 

##### Neural network ( mu ) 
- smaller the better (radians/sec) 2 -> 2.3 deg /0.02 sec so limit max to 3.

##### Increasing number of nodes in hidden layers
- 20 -> learning curve declines
- 70 -> it doesn't (Hopefully able to capture) (Need to test it out for each motion we want to train)

#### Reward Funtion (Confirmed)
- Mimic part
- Acceration small (Not necessary)
- Velocities small (Not necessary)

##### For standing motion
- Pose difference (Must)
- acceleration (Must)
- Velocities small (Very important only then it works)
- gyro rate (May be incorrect for some type of motions)


#### State Space
- All 22 joint angles
- All 22 actions taken
- gyroscope rate
- accelerometer rate
		
#### Action Space
- 22 angular velocity values (radians/sec)


### Doubts 	
- Sigma net not converging to 0. Why?
- Can we make mu span grow somehow. Coz bigger span can help only in corner cases like stopping sudden fall, but increase time of learning.
- No means to get location except using its visual receptors
- Bending backbone -> Only same lle3 and rle3 movements can make it happen. 
- Should we punish fallen state or/and include accelearation reward
- How many episodes to wait for?
- How to map progress?
- Anything Missing?
- Is moving average the best metric? What others can we use? Should see improvement/ how far from ideal
- Is cross entropy loss the best thing. How is it training the sigma network?
- Getting random trajectories for the learning curve

### TODO
- Handwave training
- Handwave training with perfect standing
	- Neural net allowed to tweak other outputs (Random restarts needed once gone to one trajectory, it doesn't explore rest of the stuff at all. How to avoid this?) 
	- Reducing action space
- Handle fallen thing? Whatever given in the checkfall.cc is wrong...

- Initial state distribution (Cannot place it in arbitrary position)
- Should include foot receptors part in states? (Difficulty as not always provided)
- Overcome Retargeting 


### Results
- Handwave difficulty as not able to learn to stand? Maybe teach standing and initialize those weights before teaching handwave
- 0.3 span it falls a lot.