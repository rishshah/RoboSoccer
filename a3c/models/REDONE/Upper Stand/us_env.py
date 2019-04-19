import sys, math
import time
import os
import numpy as np
import socket, struct
from copy import deepcopy
from client.agent import BaseAgent
from imitation.motion_clip import MotionClip
from utils import *
from Environment.utils import * 

# Getting Current Directory for default paths
CWD = os.path.dirname(__file__)
if CWD == "":
    CWD = "."

class Environment(object):
    # Global Server Constants
    TEAM = "UTAustinVilla_Base"
    U_NUM = 1
    SIMULATION_TIME = 2

    # Motion Clip Params
    MOTION_CLIP = CWD + "/imitation/mocap/stand.bvh"
    CONSTRAINTS = CWD + "/imitation/constraints/constraints_1.txt"
    SPECS = CWD + "/imitation/constraints/joint_specifications.json"
    FRAME_TIME = 0.04

    # Server and Monitor Params 
    A_PORT  = 3100
    M_PORT  = 3200
    A_HOST  = "localhost"
    SERVER  = "rcssserver3d"
    MONITOR = "rcssmonitor3d"
    LD_LIBRARY_PATH = CWD + "/server/ld_library_path"


    # Action params
    ACTION_KEYS = [
        # Hands Opposite
        # "lae1", "rae1",

        # Squats
        # "lle5", "rle5",
        # "lle4", "rle4",
        # "lle3", "rle3",

        # UpperBody
        "he1" , "he2",
        "lae1", "lae2", "lae3", "lae4",
        "rae1", "rae2", "rae3", "rae4",
        
        # Stand
        # "lle1", "lle2", "lle3", "lle4", "lle5", "lle6",
        # "rle1", "rle2", "rle3", "rle4", "rle5", "rle6",
        # "he1" , "he2",
        # "lae1", "lae2", "lae3", "lae4",
        # "rae1", "rae2", "rae3", "rae4",

        # Wave
        # "lae1", "rae1",
        # "lae2", "rae2",
        # "lae3", "rae3",
        # "lae4", "rae4",

        # WIP
        # "lle1","lle2","lle3","lle4","lle5","lle6",
        # "rle1","rle2","rle3","rle4","rle5","rle6",
        # "lae1","lae2","lae3","lae4",
        # "rae1","rae2","rae3","rae4"
    ]

    # DEFAULT_STATE_MIN = np.concatenate([np.ones(2*len(ACTION_KEYS)) * -50, np.array([-10,-10,-10, -160,-15,-15, 0])])
    # DEFAULT_STATE_RANGE = np.concatenate([np.ones(2*len(ACTION_KEYS)) * 70, np.array([5,5,5, 100,100,100, 4])])
    # DEFAULT_STATE_MIN = np.concatenate([np.ones(2*len(ACTION_KEYS)) * -50, np.array([-10,-10,-10, -160,-15,-15, -0.01,-2,-2, 0])])
    # DEFAULT_STATE_RANGE = np.concatenate([np.ones(2*len(ACTION_KEYS)) * 50, np.array([5,5,5, 100,100,100, 0.02,1,1, SIMULATION_TIME])])
    DEFAULT_STATE_MIN = np.concatenate([np.ones(2*len(ACTION_KEYS))   *-70, np.array([0])])
    DEFAULT_STATE_RANGE = np.concatenate([np.ones(2*len(ACTION_KEYS)) * 70, np.array([SIMULATION_TIME])])

    #Server Restart Parameter
    DEFAULT_ACTION = np.zeros(len(ACTION_KEYS))
    MAX_COUNT = 50

    def __init__(self,  host:str=A_HOST,  agent_port:int=A_PORT, monitor_port:int=M_PORT, motion_clip:str=MOTION_CLIP):
        self.agent_port = agent_port
        self.monitor_port = monitor_port

        self.state_dim = len(self.ACTION_KEYS)*2  + 1
        self.action_dim = len(self.ACTION_KEYS)
        
        self.agent = BaseAgent(host=host, port=agent_port, teamname=self.TEAM, player_number=self.U_NUM)
        self.motion_clip = MotionClip(mocap_file=motion_clip, constraints_file=self.CONSTRAINTS)

        self.init_time = 0
        self.count_reset = 0

        self.prev_state = None
        self.motion = []
        self.joints, self.nao_specs = get_joint_specs(self.CONSTRAINTS, self.SPECS) 
   
    def step(self, action):
        try:
            structured_action = map_action(action, self.ACTION_KEYS)             
            state, acc, gyr, pos, orr, time, is_fallen = self.agent.step(structured_action)
            tar, r = self.generate_reward(state, time, is_fallen)
            vel = get_velocity(state, self.prev_state, self.ACTION_KEYS)
            rel_time = time - self.init_time
            s = demap_state(state, acc, gyr, pos, orr, vel, tar, rel_time, self.ACTION_KEYS)            
            s = (s - self.DEFAULT_STATE_MIN)/ self.DEFAULT_STATE_RANGE         
            
            self.update_motion(state)

            return s, r, is_fallen or self.time_up(time), time        
        
        except (BrokenPipeError, ConnectionResetError, ConnectionRefusedError, socket.timeout, struct.error):
            return None, 0, True, None
    
    def generate_reward(self, state, time, is_fallen):
        target, sim = self.motion_clip.similarity(time - self.init_time, state, self.ACTION_KEYS)
        reward = np.exp(-0.003 * sim)
        print("(generate_reward) reward ", sim, reward)
        if is_fallen:
            print('(generate_reward) fallen ', time-self.init_time, np.exp(6 * (time-self.init_time)/self.SIMULATION_TIME))
            reward = np.exp(6 * (time-self.init_time)/self.SIMULATION_TIME)
        return np.array([target[s] for s in self.ACTION_KEYS]), reward

    def set_init_pose(self):
        # for i in range(56):
            # s, r, done, _ = self.step(np.array([-3.3, -2.8, 3.2, -4, -0.15, 1.27, 0, 0.33]))
            # act = np.array([3.3, 2.3, 3.2, -3.5, -0.2, 1, -0.3, 0.33])
            # s, r, done, _ = self.step(act * 26/56.0)
            # act = np.array(
            #     [0.01, 1.15, -1, -0.75, -0.7,
            #      -0.03, 0.25, -1.5, -0.75, -0.72,
            #      -4.5, -4.3, 0.5, 0.1, 
            #      -4.5, -5, 0.3, 0.1])
            # act /= 5
            # s, r, done, _ = self.step(act)
            # print("INIT_R", r, done)
        # return s
        a = np.array([
            0,#6.240369,
            0,#2.498099,
            0,#6.240369,
            0,#2.498099,
            0,# -26.441047,
            0,# 11.556906,
            0,#-8.509659,
            0,#-14.629265,
            0,#5.910771,
            0,# -34.244259,
            0,# 11.00585,
            0,#2.435798,
            -97.452478,
            -17.224523,
            -22.863067,
            -7.204805,
            -108.334258,
            14.678713,
            17.706834,
            4.102674
        ])
        for i in range(27):      
            s, r, done, _ = self.step(a/30.0)
            # print("INIT_R", r, done)
        return s
    
    def reset(self):
        self.count_reset += 1
        self.motion = []
        
        try:
            if self.count_reset == 1:
                self.cleanup()
                self.start_server()

            elif self.count_reset < self.MAX_COUNT:
                self.agent.disconnect()
            else:
                self.cleanup()
                self.start_server()
                self.count_reset = 0
        
            self.init_time = self.agent.initialize()
        
        except (BrokenPipeError, ConnectionRefusedError, ConnectionResetError, socket.timeout, struct.error):
            self.cleanup()
            self.start_server()
            self.init_time = self.agent.initialize()            

        # s = self.set_init_pose()
        # return s
        return np.zeros(self.state_dim)
        
    def cleanup(self):
        self.agent.disconnect()
        os.system("pkill -9 -f '{} --agent-port {} --server-port {}'".format(self.SERVER, self.agent_port, self.monitor_port))

    def time_up(self, time):
        if(time - self.init_time) >= self.SIMULATION_TIME:
            self.init_time = time
            return True
        else:
            return False

    def start_server(self):
        with open(self.LD_LIBRARY_PATH) as f:
            path = f.readlines()
        os.environ['LD_LIBRARY_PATH'] = path[0].strip()     
        server_command = "({} --agent-port {} --server-port {} > /dev/null 2>&1 &)".format(self.SERVER, self.agent_port, self.monitor_port)
        os.system(server_command)
        print("server starting... ")
        time.sleep(0.3)
        
    def update_motion(self, state):
        self.prev_state = deepcopy(state)
        out_list = [0,0,0]
        for j in self.joints:
            for ang in j[1:]:
                if ang in state:
                    out_list.append(state[ang])
                else:
                    out_list.append(0)
        self.motion.append(out_list)

def simulate_squats():
    env = Environment()
    env.reset()
    action = np.array([0.4, 0.4, -1.2, -1.2, 0.8, 0.8])
    for i in range(1,50):
        env.step(action)
    for i in range(1,45):
        env.step(-action)

    save_motion("./imitation/mocap/squats.bvh", env.motion, env.FRAME_TIME)

def simulate_ho():
    env = Environment()
    env.reset()
    action = [1.8, -1.8]
    for i in range(1,25):
        env.step(action)
    for i in range(1,50):
        env.step(-action)
    for i in range(1,25):
        env.step(action)
    save_motion("./imitation/mocap/hands_opposite.bvh", env.motion, env.FRAME_TIME)

if __name__ == "__main__":
    env = Environment()
    s = env.reset()
    action = env.DEFAULT_ACTION
    action[-4] = 1
    action[-8] = 1
    beta = 1
    tr = 0
    print(s[0:8])
    print(s[8:16])
    for i in range(1,200):
        # s, r, is_done, _ = env.step(action + diff * beta)
        s, r, is_done, _ = env.step(action)
        # print(i, diff, r)
        # diff = s[18:36] - s[0:18]
        # diff = s[8:16] - s[0:8]
        tr += r
        if is_done:
            break
    print(tr)
    pass
