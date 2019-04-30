import re
import numpy as np
import json

NUM_ACTIONS = 8
SIMULATION_TIME = 3.6
#Tuned Constants
DEFAULT_ACC_MIN = np.array([-10,-10,-10])
DEFAULT_ACC_RANGE = np.array([5,5,5])

DEFAULT_GYR_MIN = np.array([-100,-100])
DEFAULT_GYR_RANGE = np.array([50,50])

DEFAULT_POS_MIN = np.array([0.05])
DEFAULT_POS_RANGE = np.array([0.15])

DEFAULT_ACT_MIN = np.ones(NUM_ACTIONS) * -50
DEFAULT_ACT_RANGE = np.ones(NUM_ACTIONS) * 30

DEFAULT_VEL_MIN = np.ones(NUM_ACTIONS) * -7
DEFAULT_VEL_RANGE = np.ones(NUM_ACTIONS) * 3.5

DEFAULT_PHS_MIN = np.array([0])
DEFAULT_PHS_RANGE = np.array([SIMULATION_TIME/2])

DEFAULT_STATE_MIN = np.concatenate([DEFAULT_ACT_MIN, DEFAULT_VEL_MIN, DEFAULT_ACC_MIN, DEFAULT_GYR_MIN, DEFAULT_POS_MIN, DEFAULT_PHS_MIN])
DEFAULT_STATE_RANGE = np.concatenate([DEFAULT_ACT_RANGE, DEFAULT_VEL_RANGE, DEFAULT_ACC_RANGE, DEFAULT_GYR_RANGE, DEFAULT_POS_RANGE, DEFAULT_PHS_RANGE])

def save_motion(file, motion, ftime):
    with open(file, 'a') as f:
        f.write("Frames: " + str(len(motion)) + "\n")    
        f.write("Frame Time: " + str(ftime) + "\n")    
        for frame in motion:
            frame = [str(a) for a in frame]
            f.write(" ".join(frame) + "\n")

def get_velocity(state, prev_state, action_keys):
    if prev_state:
        return np.array([state[s] - prev_state[s] for s in action_keys])
    else:
        return np.zeros(len(action_keys))

def map_action(action, action_keys):
    tmp = {}
    for i, s in enumerate(action_keys):
        tmp[s] = action[i]
    # print("(map_action) ", tmp)
    return tmp

def demap_state(state, acc, gyr, pos, orr, velocities, target, rel_time, action_keys):
    tmp = [state[s]for s in action_keys]
    # tmp += list(target)
    tmp += list(velocities)
    tmp += list(acc)
    tmp += list(gyr)
    tmp += list([pos])
    # tmp += [orr]
    tmp += list([rel_time])
    x = (np.array(tmp) - DEFAULT_STATE_MIN)/ DEFAULT_STATE_RANGE         
    return x         
    # return np.array(tmp)         

def get_joint_specs(constrains_path, specs_path):
    with open(constrains_path, 'r') as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    joints = [list(filter(None, c.split('\t'))) for c in content]

    with open(specs_path, 'r') as f:
        content = f.readlines()
    json_str = re.sub('\s+', '', "".join(content))
    return joints, json.loads(json_str)

def calc_com(curr_frame, specs):
    for joint in specs:
        ang, axis = curr_frame[joint["orr"]], joint["orr"] % 3

