# Skills

## Hands Opposite Motion
- Allowed Joints 2
- Relu Activation everywhere
- lr 0.001
- 1 thread

### State Space
- 2 joints
- 2 Velocities
- 3 GYR
- 3 ACC
- 3 Pos
- 1 Orr

### Reward Function
- -690
- Sum of square of all joint position differences from time-linear motion of hands  
```
 x = (state["lae1"] + time * 20)* (state["lae1"] + time * 20)
 x += (state["rae1"] + time * 20)* (state["rae1"] + time * 20)
 reward = 10 - 0.1 * x
```


## Stay steady 
- Allowed Joints 12 ( 4 lefthand + 4 righthand + 2 head + 2 knees)
- Tanh Activation everywhere
- lr 0.001
- 2 threads

### State Space
- 12 joints
- 12 Velocities
- 3 GYR
- 3 ACC
- 3 Pos
- 1 Orr
- 1 Time

### Reward Function
- -1590
- Sum of square of all joint position differences
```
	x = sum([state[s]*state[s] for s in state])
    reward = 10 - 0.1 * x
```


## Situps 
- Allowed Joints 4 ( 2 hips + 2 knees)
- Tanh Activation everywhere
- lr 0.001

### State Space
- 4 joints
- 4 Velocities
- 3 GYR
- 3 ACC
- 3 Pos
- 1 Orr
- 1 Time

### Reward Function 
- Sum of square of all joint position differences

### Hazards
- Not letting it learn to fall quickly so as to get fewer negative rewards in longer run
- Fallen punishment ? Must be large negative reward



