import sys, math
import time
import os
import numpy as np
import socket, struct
from client.agent import BaseAgent
from imitation.motion_clip import MotionClip

# Getting Current Directory for default paths
CWD = os.path.dirname(__file__)
if CWD == "":
    CWD = "."

class Environment(object):
    # Global Server Constants
    TEAM = "UTAustinVilla_Base"
    U_NUM = 1
    SIMULATION_TIME = 4

    # Motion Clip Params
    MOTION_CLIP = CWD + "/imitation/debug/situps.bvh"
    CONSTRAINTS = CWD + "/imitation/constraints.txt"
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

        # UpperBody + knees
        # "lle4", "rle4",
        # "he1","he2",
        # "lae1","lae2","lae3","lae4",
        # "rae1","rae2","rae3","rae4"
        
        # Stand
        # "lle1","lle2","lle3","lle4","lle5","lle6",
        # "rle1","rle2","rle3","rle4","rle5","rle6",
        # "he1","he2",
        # "lae1","lae2","lae3","lae4",
        # "rae1","rae2","rae3","rae4"
        
        # Situps
        "lle5", "rle5",
        "lle4", "rle4",
        "lle3", "rle3",
    ]

    DEFAULT_ACTION = np.zeros(len(ACTION_KEYS));    
    DEFAULT_STATE_MIN = np.concatenate([np.ones(3*len(ACTION_KEYS)) * -50, np.array([-10,-10,-10, -160,-15,-15, 0])])
    DEFAULT_STATE_RANGE = np.concatenate([np.ones(3*len(ACTION_KEYS)) * 50, np.array([5,5,5, 100,100,100, SIMULATION_TIME])])
    # DEFAULT_STATE_MIN = np.concatenate([np.ones(3*len(ACTION_KEYS)) * -50, np.array([-10,-10,-10, -160,-15,-15, -0.01,-2,-2, 80, 0])])
    # DEFAULT_STATE_RANGE = np.concatenate([np.ones(3*len(ACTION_KEYS)) * 50, np.array([5,5,5, 100,100,100, 0.02,1,1, 100, SIMULATION_TIME])])
    # DEFAULT_STATE_MIN = np.concatenate([np.ones(3*len(ACTION_KEYS)) * -50, np.array([0])])
    # DEFAULT_STATE_RANGE = np.concatenate([np.ones(3*len(ACTION_KEYS)) * 50, np.array([SIMULATION_TIME])])

    #Server Restart Parameter
    MAX_COUNT = 50

    def __init__(
        self, 
        host:str=A_HOST, 
        agent_port:int=A_PORT,
        monitor_port:int=M_PORT,
        motion_clip:str=MOTION_CLIP):

        self.agent_port = agent_port
        self.monitor_port = monitor_port

        self.state_dim = len(self.ACTION_KEYS)*3  + 7
        self.action_dim = len(self.ACTION_KEYS)
        
        self.agent = BaseAgent(host=host, port=agent_port, teamname=self.TEAM, player_number=self.U_NUM)
        self.motion_clip = MotionClip(mocap_file=motion_clip, constraints_file=self.CONSTRAINTS)

        self.init_time = 0
        self.count_reset = 0

        # For saving handmade motions
        with open(self.CONSTRAINTS, 'r') as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        self.joints = [list(filter(None, c.split('\t'))) for c in content]
        self.motion = [] 

        self.prev_state = {} # For velocities
   
    def get_velocity(self, state):
        if self.prev_state:
            tmp = [state[s] - self.prev_state[s] for s in self.ACTION_KEYS]
            self.prev_state = state
            return np.array(tmp)
        else:
            self.prev_state = state
            return self.DEFAULT_ACTION 

    def map_action(self, action):
        tmp = {}
        for i, s in enumerate(self.ACTION_KEYS):
            tmp[s] = action[i]
        # print(tmp)
        return tmp

    def demap_state(self, state, acc, gyr, pos, orr, velocities, target, time):
        tmp = [state[s]for s in self.ACTION_KEYS]
        tmp = tmp + list(velocities)
        tmp = tmp + list(target)
        tmp = tmp + list(acc)
        tmp = tmp + list(gyr)
        # tmp = tmp + list(pos)
        # tmp = tmp + [orr]
        tmp = tmp + [time - self.init_time - self.FRAME_TIME]  
        tmp = (np.array(tmp) - self.DEFAULT_STATE_MIN)/ self.DEFAULT_STATE_RANGE         
        return tmp

    def step(self, action):
        try:
            state, acc, gyr, pos, orr, time, is_fallen = self.agent.step(self.map_action(action))
            self.update_motion(state)
            target, r = self.generate_reward(state, time, is_fallen)  
            s = self.demap_state(state, acc, gyr, pos, orr, self.get_velocity(state), target, time)
            
            return s, r, is_fallen or self.time_up(time), None        
        
        except (BrokenPipeError, ConnectionResetError, ConnectionRefusedError, socket.timeout, struct.error):
            return None, 0, True, None
    
    def generate_reward(self, state, time, is_fallen):
        target, sim = self.motion_clip.similarity(time - self.init_time, state, self.ACTION_KEYS)
        reward = -0.001 * sim
        # print("(generate_reward)", reward)
        if is_fallen:
            print('(generate_reward) fallen ', time-self.init_time)
            reward -= 1000 * (1 - (time-self.init_time)/self.SIMULATION_TIME)
        # print("TARGET:", target)
        return np.array([target[s] for s in self.ACTION_KEYS]), reward

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

        return np.zeros(self.state_dim)
        
    def cleanup(self):
        self.agent.disconnect()
        os.system("pkill -9 -f '{} --agent-port {} --server-port {}'".format(self.SERVER, self.agent_port, self.monitor_port));

    def time_up(self, time):
        return (time - self.init_time) >= self.SIMULATION_TIME

    def start_server(self):
        with open(self.LD_LIBRARY_PATH) as f:
            path = f.readlines()
        os.environ['LD_LIBRARY_PATH'] = path[0].strip()     
        server_command = "({} --agent-port {} --server-port {} > /dev/null 2>&1 &)".format(self.SERVER, self.agent_port, self.monitor_port)
        os.system(server_command)
        print("server starting... ")
        time.sleep(0.3);
        
    def update_motion(self, state):
        out_list = [0,0,0]
        for j in self.joints:
            for ang in j[1:]:
                if ang in state:
                    out_list.append(state[ang])
                else:
                    out_list.append(0)
        self.motion.append(out_list)

    def save_motion(self, file):
        with open(file, 'a') as f:
            f.write("Frames: " + str(len(self.motion)) + "\n")    
            f.write("Frame Time: self.FRAME_TIME\n")    
            for frame in self.motion:
                frame = [str(a) for a in frame]
                f.write(" ".join(frame) + "\n")


if __name__ == "__main__":
    # env = Environment()
    # env.reset()
    # action = env.DEFAULT_ACTION;
    # action[0] = 0.1
    # action[1] = 0.1
    # action[2] = -0.3;
    # action[3] = -0.3;
    # action[4] = 0.2;
    # action[5] = 0.2;
    # for i in range(1,300):
    #     env.step(action);

    # action[0] = -0.1
    # action[1] = -0.1
    # action[2] = 0.3;
    # action[3] = 0.3;
    # action[4] = -0.2;
    # action[5] = -0.2;
    # for i in range(1,300):
    #     env.step(action);

    # env.save_motion("./imitation/debug/situps.bvh")


    env = Environment()
    env.reset()
    action = env.DEFAULT_ACTION;
    action[0] = 0.4;
    action[1] = -0.4;
    for i in range(1,100):
        env.step(action);

    action[0] = -0.4;
    action[1] = 0.4;
    for i in range(1,200):
        env.step(action);

    action[0] = 0.4;
    action[1] = -0.4;
    for i in range(1,100):
        env.step(action);

    env.save_motion("./imitation/debug/hands_opposite.bvh")

    # env = Environment()
    # env.reset()
    # action = env.DEFAULT_ACTION;
    # for i in range(1,500):
    #     env.step(action);

    # env.save_motion("./imitation/debug/stand.bvh")
    pass
