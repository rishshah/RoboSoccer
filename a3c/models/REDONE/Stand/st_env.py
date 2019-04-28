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
        # Hands Opposite DONE
        # "lae1", "rae1",

        # Squats INP
        # "lle2", "rle2",
        # "lle5", "rle5",
        # "lle4", "rle4",
        # "lle3", "rle3",

        # UpperBody DONE
        # "he1" , "he2",
        # "lae1", "lae2", "lae3", "lae4",
        # "rae1", "rae2", "rae3", "rae4",
        
        # Stand
        "lle1", "lle2", "lle3", "lle4", "lle5", "lle6",
        "rle1", "rle2", "rle3", "rle4", "rle5", "rle6",
        "he1" , "he2",
        "lae1", "lae2", "lae3", "lae4",
        "rae1", "rae2", "rae3", "rae4",

        # Wave
        # "lae1", "rae1",
        # "lae2", "rae2",
        # "lae3", "rae3",
        # "lae4", "rae4",

        # WIP PARTIAL
        # "lle1","lle2","lle3","lle4","lle5","lle6",
        # "lle1","rle2","rle3","rle4","rle5","rle6",
    ]

    #Server Restart Parameter
    DEFAULT_ACTION = np.zeros(len(ACTION_KEYS))
    MAX_COUNT = 50

    #Reward Hyperparams
    COPY = -0.04/len(ACTION_KEYS)
    # FALLEN = 0.5
    HEIGHT = -0.1
    HEIGHT_THRESHOLD = 0.5 
    # GYR = 0.00005
    GYR_THRESHOLD = 90

    def __init__(self,  host:str=A_HOST,  agent_port:int=A_PORT, monitor_port:int=M_PORT, motion_clip:str=MOTION_CLIP):
        self.agent_port = agent_port
        self.monitor_port = monitor_port

        self.state_dim = len(self.ACTION_KEYS)*2 + 7
        self.action_dim = len(self.ACTION_KEYS)
        
        self.agent = BaseAgent(host=host, port=agent_port, teamname=self.TEAM, player_number=self.U_NUM)
        self.motion_clip = MotionClip(mocap_file=motion_clip, constraints_file=self.CONSTRAINTS)

        self.init_time = 0
        self.count_reset = 0

        self.prev_state = None
        self.motion = []
        self.joints, self.nao_specs = get_joint_specs(self.CONSTRAINTS, self.SPECS) 
   
    def step(self, action, t=None):
        try:
            state, acc, gyr, pos, orr, time, is_fallen = self.agent.step(map_action(action, self.ACTION_KEYS))
            rel_time = time - self.init_time
            tar, r = self.generate_reward(state, gyr[0:2], pos[-1], acc, rel_time, is_fallen, t)

            vel = get_velocity(state, self.prev_state, self.ACTION_KEYS)            
            s = demap_state(state, acc, gyr[0:2], pos[-1], orr, vel, tar, rel_time, self.ACTION_KEYS)            
            
            self.update_motion(state)
            
            # For Reset phase for periodic motions
            done = False
            if is_fallen or self.time_up(time):
                self.init_time = time
                done = True

            return s, r, done, time        
        
        except (BrokenPipeError, ConnectionResetError, ConnectionRefusedError, socket.timeout, struct.error):
            return None, 0, True, None
    
    def generate_reward(self, state, gyr, pos, acc, time, is_fallen, t=None):
        copy_reward, gyr_reward, pos_reward, fallen_reward = 0,0,0,0
        
        target, sim = self.motion_clip.similarity(max(0,time - self.FRAME_TIME), state, self.ACTION_KEYS)
        copy_reward = np.exp(self.COPY * sim)
        
        if math.fabs(gyr[0]) < self.GYR_THRESHOLD and math.fabs(gyr[1]) < self.GYR_THRESHOLD:
            gyr_reward = 0.05 # np.exp(self.GYR * (gyr[0]**2 + gyr[1]**2))

        pos = max(pos, 0.01)
        if pos > self.HEIGHT_THRESHOLD:
            pos_reward = 0.3 * np.exp(self.HEIGHT * (1/pos))
        
        if(t != None and t % 30 == 0):
            print("(generate_reward) {} \t (cpy, [{},{}]), (acc, {})\t (gyr, [{},{}]), \t (pos, [{},{}])".format(
                round(time,2), round(sim,2), round(copy_reward,2), acc, gyr, round(gyr_reward,2), pos, round(pos_reward,2)))

        if is_fallen:
            # fallen_reward = np.exp(self.FALLEN * self.SIMULATION_TIME/time)
            print('(generate_reward) fallen ', time)

        reward = copy_reward + gyr_reward + pos_reward + fallen_reward
        return np.array([target[s] for s in self.ACTION_KEYS]), reward

    def set_init_pose(self, max_steps, num_steps):
        conversion_factor = 180/np.pi
        conversion_factor /= 50

        init_state = map_action(self.DEFAULT_ACTION, self.ACTION_KEYS)
        target, sim = self.motion_clip.similarity(0, init_state, self.ACTION_KEYS)
        diff = get_velocity(target, init_state, self.ACTION_KEYS)
        diff /= conversion_factor
        for i in range(num_steps):      
            s, r, done, time = self.step(diff/max_steps)
            self.init_time = time 
            print(i, "INIT_R", r, done)
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

        # Stabilizing the intiial state
        for i in range(1,40):
            s, _, _, time = self.step(self.DEFAULT_ACTION)
            self.init_time = time 
        
        # return self.set_init_pose(60, 30)
        return s
        
    def cleanup(self):
        self.agent.disconnect()
        os.system("pkill -9 -f '{} --agent-port {} --server-port {}'".format(self.SERVER, self.agent_port, self.monitor_port))

    def time_up(self, time):
        return (time - self.init_time) >= self.SIMULATION_TIME

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
    action = np.array([0, 0, 0.4, 0.4, -1.2, -1.2, 0.8, 0.8])
    tr = 0
    for i in range(1,50):
        _, r, _, _ = env.step(action)
        tr += r
    for i in range(1,45):
        env.step(-action)
        tr += r
    print(tr)
    env.cleanup()
    # save_motion("./imitation/mocap/squats.bvh", env.motion, env.FRAME_TIME)

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
    env.cleanup()
    # save_motion("./imitation/mocap/hands_opposite.bvh", env.motion, env.FRAME_TIME)

def simulate_stand():
    env = Environment()
    env.reset()
    action = env.DEFAULT_ACTION
    for i in range(1,100):
        env.step(action)
    env.cleanup()

def simulate_fall():
    env = Environment()
    env.reset()
    action = [0,0, 0.4, 0.4, 0,0, 0,0]
    for i in range(1,200):
        env.step(action)
    env.cleanup()

def simulate_given():
    env = Environment()
    s = env.reset()
    action = env.DEFAULT_ACTION
    tr = 0
    beta = 1
    for i in range(1,80):
        s, r, is_done, _ = env.step(action)
        diff = s[env.action_dim:2*env.action_dim] - s[0:env.action_dim]
        action = np.clip(beta * diff, -3,3)
        print(diff)
        tr += r
        # if is_done:
        #     break
    print(tr)
    env.cleanup()    

if __name__ == "__main__":
    simulate_given()
