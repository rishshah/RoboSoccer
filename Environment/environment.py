import sys
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
    SIMULATION_TIME = 7.9

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

    DIM = len(STATE_KEYS)
    DEFAULT_ACTION = np.zeros(len(STATE_KEYS));
    
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

        self.state_dim = self.DIM*2  + 11
        self.action_dim = self.DIM
        
        self.agent = BaseAgent(host=host, port=agent_port, teamname=self.TEAM, player_number=self.U_NUM)
        self.motion_clip = MotionClip(mocap_file=motion_clip, constraints_file=self.CONSTRAINTS)

        self.count_reset = 0
        self.time = 0

        with open(self.CONSTRAINTS, 'r') as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        self.joints = [list(filter(None, c.split('\t'))) for c in content]
        
        self.motion = []
        
    def map_action(self, action, sp=False):
        tmp = {}
        for i, s in enumerate(self.STATE_KEYS):
            tmp[s] = action[i]
        # print(tmp)
        return tmp

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

    def demap_state(self, state, acc, gyr, pos, orr, velocities, time):
        tmp = [state[s]/10.0 for s in self.STATE_KEYS]
        tmp = tmp + list([5*v for v in velocities])
        tmp = tmp + list(acc)
        tmp = tmp + list([10*g for g in gyr])
        tmp = tmp + list([6*p for p in pos])
        tmp = tmp + [orr/10]
        tmp = tmp + [(time - self.time)*2]          
        return np.array(tmp)        

    def step(self, action, sp=False):
        try:
            state, acc, gyr, pos, orr, time, is_fallen = self.agent.step(self.map_action(action, sp))
            self.update_motion(state)

            s = self.demap_state(state, acc, gyr, pos, orr, action, time)
            r = self.generate_reward(state, acc, gyr, action, time, is_fallen)  
            return s, r, self.time_up(time), None        
        
        except (BrokenPipeError,ConnectionResetError):
            return None, 0, True, None
    
    def generate_reward(self, state, acc, gyr, action, time, is_fallen):
        sim = self.motion_clip.similarity(time - self.time, state, self.STATE_KEYS)
        # print(sim)
        reward = 10 + sim
        # if is_fallen:
        #     print('(generate_reward) fallen ', acc)
        #     reward -= 100000
        # elif self.time_up(time):
        #     reward += 10000
        return reward * 0.01

    def reset(self):
        self.count_reset += 1
        self.motion = []
        
        try:
            if self.count_reset == 1:
                self.cleanup()
                self.agent.disconnect()
                self.start_server()
                self.time = self.agent.initialize()

            elif self.count_reset < self.MAX_COUNT:
                self.agent.disconnect()
                self.time = self.agent.initialize()
            else:
                self.cleanup()
                self.start_server()
                self.time = self.agent.initialize()
                self.count_reset = 0
        
        
        except (ConnectionRefusedError, ConnectionResetError, socket.timeout, struct.error):
            self.cleanup()
            self.start_server()
            self.time = self.agent.initialize()            

        state, _, _, _ = self.step(action=self.DEFAULT_ACTION)
        return state
        
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


    # env = Environment()
    # env.reset()
    # action = env.DEFAULT_ACTION;
    # action[0] = 0.4;
    # action[1] = -0.4;
    # for i in range(1,100):
    #     env.step(action);

    # action[0] = -0.4;
    # action[1] = 0.4;
    # for i in range(1,200):
    #     env.step(action);

    # action[0] = 0.4;
    # action[1] = -0.4;
    # for i in range(1,100):
    #     env.step(action);

    # env.save_motion("./imitation/debug/hands_opposite.bvh")

    # env = Environment()
    # env.reset()
    # action = env.DEFAULT_ACTION;
    # for i in range(1,500):
    #     env.step(action);

    # env.save_motion("./imitation/debug/stand.bvh")
    pass
