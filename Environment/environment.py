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
    # Global Constants
    TEAM = "UTAustinVilla_Base"
    U_NUM = 1
    SIMULATION_TIME = 4

    #Connection to Server and Motion Clip
    MOTION_CLIP = CWD + "/imitation/debug/hands_opposite.bvh"
    CONSTRAINTS = CWD + "/imitation/constraints.txt"
    
    # Server and Monitor
    A_PORT  = 3100
    M_PORT  = 3200
    A_HOST  = "localhost"
    SERVER  = "rcssserver3d"
    MONITOR = "rcssmonitor3d"
    LD_LIBRARY_PATH = CWD + "/server/ld_library_path"
    
    # Details of state and action parameters
    STATE_KEYS = [
        # Hands Opposite
        "lae1", "rae1",

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
        # "lle5", "rle5",
        # "lle4", "rle4",
        # "lle3", "rle3",
    ]

    A_DIM = len(STATE_KEYS)
    DEFAULT_ACTION = np.zeros(len(STATE_KEYS));
    DEFAULT_STATE_MIN = np.concatenate([np.ones(3*len(STATE_KEYS)) * -100, np.array([0])])
    DEFAULT_STATE_RANGE = np.concatenate([np.ones(3*len(STATE_KEYS)) * 100, np.array([4])])
    # DEFAULT_STATE_MIN = np.concatenate([np.ones(3*len(STATE_KEYS)) * -100, np.array([-10,-10,-10, -160,-15,-15, -0.01,-2,-2, 80, 0])])
    # DEFAULT_STATE_RANGE = np.concatenate([np.ones(3*len(STATE_KEYS)) * 100, np.array([5,5,5, 150,150,150, 0.02,1,1, 100, 4])])

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

        self.state_dim = self.A_DIM*3  + 1
        self.action_dim = self.A_DIM
        
        self.agent = BaseAgent(host=host, port=agent_port, teamname=self.TEAM, player_number=self.U_NUM)
        self.motion_clip = MotionClip(mocap_file=motion_clip, constraints_file=self.CONSTRAINTS)

        self.count_reset = 0
        self.time = 0
        self.xxtime = 0
        with open(self.CONSTRAINTS, 'r') as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        self.joints = [list(filter(None, c.split('\t'))) for c in content]
        
        self.motion = []
        self.prev_state = {}
   
    def get_velocity(self, state):
        if self.prev_state:
            tmp = [state[s] - self.prev_state[s] for s in self.STATE_KEYS]
            self.prev_state = state
            return np.array(tmp)
        else:
            self.prev_state = state
            return self.DEFAULT_ACTION 

    def map_action(self, action):
        tmp = {}
        for i, s in enumerate(self.STATE_KEYS):
            tmp[s] = action[i]
        # if int(math.fabs(self.xxtime - self.time)*3) % 10 == 0:  
        # print(tmp)
        return tmp

    def demap_state(self, state, acc, gyr, pos, orr, velocities, target, time):
        tmp = [state[s]for s in self.STATE_KEYS]
        tmp = tmp + list(velocities)
        tmp = tmp + list(target)
        # tmp = tmp + list(acc)
        # tmp = tmp + list(gyr)
        # tmp = tmp + list(pos)
        # tmp = tmp + [orr]
        tmp = tmp + [time - self.time - 0.02]  
        tmp = (np.array(tmp) - self.DEFAULT_STATE_MIN)/ self.DEFAULT_STATE_RANGE         
        return tmp

    def step(self, action):
        try:
            state, acc, gyr, pos, orr, time, is_fallen = self.agent.step(self.map_action(action))
            self.update_motion(state)
            target, r = self.generate_reward(state, time, is_fallen)  
            s = self.demap_state(state, acc, gyr, pos, orr, self.get_velocity(state), target, time)
            self.xxtime = time
            
            return s, r, is_fallen or self.time_up(time), None        
        
        except (BrokenPipeError, ConnectionResetError, ConnectionRefusedError, socket.timeout, struct.error):
            return None, 0, True, None
    
    def generate_reward(self, state, time, is_fallen):
        target, sim = self.motion_clip.similarity(time - self.time, state, self.STATE_KEYS)
        # reward = 10 * np.exp(-0.001 * sim)
        # reward = 10 * 1.0/(1 + 0.001 * sim)
        reward = -0.0005 * sim
        # print("(generate_reward)", reward)
        # print(reward)
        # if is_fallen:
        #     print('(generate_reward) fallen ', time-self.time)
        #     reward -= 1000 * np.exp(-(time - self.time)/20)
        # print(reward)
        # print("(generate_reward)", target)
        return np.array([target[s] for s in self.STATE_KEYS]), reward

    def reset(self):
        self.count_reset += 1
        self.motion = []
        self.xxtime = 0
        
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
        
            self.time = self.agent.initialize()
        
        except (BrokenPipeError, ConnectionRefusedError, ConnectionResetError, socket.timeout, struct.error):
            self.cleanup()
            self.start_server()
            self.time = self.agent.initialize()            

        # state, _, _, _ = self.step(action=self.DEFAULT_ACTION)
        # print("(Reset)", state)
        # print("(Reset)", type(state))
        return np.concatenate([np.zeros(3*len(self.STATE_KEYS)), np.array([0])])
        
    def cleanup(self):
        self.agent.disconnect()
        os.system("pkill -9 -f '{} --agent-port {} --server-port {}'".format(self.SERVER, self.agent_port, self.monitor_port));

    def time_up(self, time):
        return (time - self.time) >= self.SIMULATION_TIME

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
            f.write("Frame Time: 0.02\n")    
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

    env.save_motion("./imitation/debug/hands_opposite_tmp1.bvh")

    # env = Environment()
    # env.reset()
    # action = env.DEFAULT_ACTION;
    # for i in range(1,500):
    #     env.step(action);

    # env.save_motion("./imitation/debug/stand.bvh")
    pass
