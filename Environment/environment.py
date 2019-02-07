import sys
import time
import os
import numpy as np
import socket, struct
from client.agent import BaseAgent
from imitation.motion_clip import MotionClip

CWD = os.path.dirname(__file__)

class Environment(object):
    # Global Constants
    TEAM = "UTAustinVilla_Base"
    U_NUM = 1
    SIMULATION_TIME = 2

    #Connection to Server and Motion Clip
    A_HOST = "localhost"
    A_PORT = 3100
    M_PORT = 3200
    MOTION_CLIP = CWD + "/imitation/ss.bvh"
    CONSTRAINTS = CWD + "/imitation/constraints.txt"
    
    #Server and Monitor
    LD_LIBRARY_PATH = CWD + "/server/ld_library_path"
    SERVER = "rcssserver3d"
    MONITOR = "rcssmonitor3d"
    
    # Details of state and action parameters
    STATE_KEYS = [
        # "lle1","lle2","lle3","lle4","lle5","lle6",
        # "rle1","rle2","rle3","rle4","rle5","rle6",
        # "lle4", "rle4",
        "he1","he2",
        "lae1",#"lae2","lae3","lae4",
        "rae1",#"rae2","rae3","rae4"
    ]

    DIM = len(STATE_KEYS)
    DEFAULT_ACTION = np.zeros(len(STATE_KEYS));

    #Tuning Parameters
    ALPHA = 10
    BETA = 0.1
    GAMMA = 1
    
    #Server and Monitor Restart Parameters
    MAX_COUNT = 50

    def __init__(
        self, 
        host:str=A_HOST, 
        agent_port:int=A_PORT,
        monitor_port:int=M_PORT,
        motion_clip:str=MOTION_CLIP):

        self.agent_port = agent_port
        self.monitor_port = monitor_port
        self.host = host

        self.state_dim = self.DIM*2  + 10
        self.action_dim = self.DIM
        
        self.agent = BaseAgent(host=host, port=agent_port, teamname=self.TEAM, player_number=self.U_NUM)
        self.motion_clip = MotionClip(mocap_file=motion_clip, constraints_file=self.CONSTRAINTS)

        self.count_reset = 0
        self.time = 0
        self.max_acc_x = 0
        self.min_acc_x = 0
        self.max_acc_y = 0
        self.min_acc_y = 0

    def map_action(self, action, sp=False):
        tmp = {}
        for i, s in enumerate(self.STATE_KEYS):
            tmp[s] = action[i]
            # if "ll" in s or "rl" in s:
            #     tmp[s] = 0
        return tmp

    def demap_state(self, state, acc, gyr, pos, orr, velocities):
        tmp = [state[s] for s in self.STATE_KEYS]
        tmp = tmp + list(velocities)
        tmp = tmp + list(acc)
        tmp = tmp + list(gyr)
        tmp = tmp + list(pos)
        tmp = tmp + [orr]
        return np.array(tmp)        

    def step(self, action, sp=False):
        try:
            state, acc, gyr, pos, orr, time, is_fallen = self.agent.step(self.map_action(action, sp))
            ret_state = self.demap_state(state, acc, gyr, pos, orr, action)
            ret_reward = self.generate_reward(state, acc, gyr, action, time, is_fallen)  
            return ret_state, ret_reward, self.time_up(time), None        
        
        except (BrokenPipeError,ConnectionResetError):
            return None, 0, True, None

    def reward_state(self, state, time):
        x = sum([state[s]*state[s] for s in state])
        x -= state["lae1"]*state["lae1"] + state["rae1"]*state["rae1"]
        x = (state["lae1"] + time * 20)* (state["lae1"] + time * 20)
        x += (state["rae1"] + time * 20)* (state["rae1"] + time * 20)
        # print("(reward_state) ", time, state["rae1"], (state["rae1"] - time * 20), x)
        return -0.1*x
    
    def generate_reward(self, state, acc, gyr, action, time, is_fallen):
        # x = self.motion_clip.similarity(time=time - self.time, actual_pose=state, keys=self.STATE_KEYS)
        # reward_acc = -0.01 * (acc[0]*acc[0] + acc[1]*acc[1] + acc[2]*acc[2]) 
        # print("reward_acc", reward_acc)
        # reward_vel = -0 * sum([s*s for s in action]); 
        # print("reward_vel", reward_vel)
        self.max_acc_y = max(self.max_acc_y, acc[1])
        self.min_acc_y = min(self.min_acc_y, acc[1])
        self.max_acc_x = max(self.max_acc_x, acc[0])
        self.min_acc_x = min(self.min_acc_x, acc[0])
                
        reward = 10 + self.reward_state(state, time - self.time) 
        # if is_fallen:
        #     print('(generate_reward) fallen ', acc)
        #     reward -= 100000
        # elif self.time_up(time):
            # reward .+= 100000
        # print(reward)
        return reward

    def reset(self):
        self.count_reset += 1
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

        state, reward, done, _ = self.step(action=self.DEFAULT_ACTION)
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
        time.sleep(1);
        


if __name__ == "__main__":
    env = Environment(agent_port=3501, monitor_port=3502)
    env.reset()
    action = np.zeros(6);
    action[0] = 0.1;
    action[1] = 0.1;
    for i in range(1,1000):
        env.step(action);
