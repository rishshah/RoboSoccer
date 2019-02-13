# Skills

## Hands Opposite Motion
- Allowed Joints 2
- Tanh Activation everywhere
- lr 0.001
- 3 thread
- hand_opposite.bvh

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


## Stay steady 
- Allowed Joints 12 ( 4 lefthand + 4 righthand + 2 head + 2 knees)
- Tanh Activation everywhere
- lr 0.001
- 2 threads
- stand.bvh

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


## Situps 
- Allowed Joints 6 ( 2 hips + 2 knees + 2 ankles)
- Tanh Activation everywhere
- lr 0.001
- situps.bvh

### State Space
- 4 joints
- 4 Velocities
- 3 GYR
- 3 ACC
- 3 Pos
Tanh 1 Orr
- 1 Time

### Reward Function 
- Sum of square of all joint position differences

### Hazards
- Not letting it learn to fall quickly so as to get fewer negative rewards in longer run
- Fallen punishment ? Must be large negative reward



