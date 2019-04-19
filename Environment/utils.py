import re
import numpy as np
import json

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
    # tmp += tmp + list(velocities)
    tmp += list(target)
    # tmp += list(acc)
    # tmp += list(gyr)
    # tmp += list(pos)
    # tmp += [orr]
    tmp += [rel_time]  
    return np.array(tmp)

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

