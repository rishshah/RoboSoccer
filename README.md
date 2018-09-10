# RoboSoccer
Teaching elementary skills to robots using motion clips and Reinforcement Learning

## Status

### Done
- Clips found to be used

### TODO

#### Part1
- Read remaining parts of paper to figure out type of neural network used
- Check if simulator is there in the paper itself 
- Find simulator
- Training the neural network
- Figure out all parts of MDP and create environment for RL

#### Part2
- Retargetting the clips (if necessary) 
- Figure out simulation environment (Run Aston Villa simulation (without bugs))
- Are torques required?
- Incorporating trained controllers in the existing framework

#### Part3
- Figure out the reward functions for goal oriented tasks

## Decoding the paper

#### States
- Relative position of all links, their orientation in quaternions, linear and angular velocities.
- Time variable “phase” between 0 and 1
- Cyclic motions have that phase set to 0 at end

#### Actions
- Target orientation at each joints

#### Network
- Google how to use RL in neural net

#### Reward

- Imitation reward
-- Pose reward (difference between join orientations)
-- Velocity reward (difference between velocities)
-- End effector reward
-- Center of mass deviation penalty 

- Goal reward
-- LATER

#### Training
- Use of PPO with Clipped surrogate objective

#### Two networks 
- Value function
- Policy

#### Initial state distribution
Start in all possible states in the clip rather than at the start as there is a chance that high reward states are present in the very end

#### Early termination
If body torso  or head or some other links hit the ground then terminate the learning as it is wasted in getting up from those positions



### Discussion
- Nao model created (check if correct)
	-- Some Joints have been assumed not sure what to do ? SHIVARAM
	-- Which joint in effector maps to which joint in model is difficult ? PARAG 
	-- We would need that ?
- What to do with those rsg files? SHIVARAM
- Constrained nao cannot be retargetted to , so cannot test the files via skills ? BOTH
- Simulator working 
- In process of understanding A3C
