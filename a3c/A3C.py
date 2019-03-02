import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import math, os
import numpy as np

sys.path.append('../')
sys.path.append('../Environment')

from utils import v_wrap, set_init, push_and_pull, record
from shared_adam import SharedAdam
from Environment.environment import Environment
import copy 

# Global Variables and HyperParameters
NUM_THREADS = 2
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)

env = Environment()
N_S = env.state_dim
N_A = env.action_dim

GAMMA = 1
MAX_EP = 3000
MAX_EP_STEP = 500
Z = 250 

# modelName = 'upper_body+2legjoint.pt'
# modelName = 'hand_opposite_net.pt'
# modelName = 'stand.pt'
# modelName = 'falling_but_following.pt'
modelName = 'int_net.pt'
loadModel = False
testModel = False
learning_rate = 0.00002

is_gpu_available = torch.cuda.is_available()
class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        
        self.com1 = nn.Linear(s_dim, Z)
        self.com2 = nn.Linear(Z, Z)
        
        self.mu1 = nn.Linear(Z, Z)
        self.sigma1 = nn.Linear(Z, Z)
        self.mu = nn.Linear(Z, a_dim)
        self.sigma = nn.Linear(Z, a_dim)
        

        self.c1 = nn.Linear(s_dim, Z)
        self.c2 = nn.Linear(Z, Z)
        self.c3 = nn.Linear(Z, Z)
        self.v = nn.Linear(Z, 1)
        
        set_init([self.com1, self.com2, self.mu1, self.sigma1, self.mu, self.sigma, self.c1, self.c2, self.c3, self.v])
        self.distribution = torch.distributions.Normal

    def forward(self, x):
        com1 = F.relu6(self.com1(x))
        com2 = F.relu6(self.com2(com1))
        
        mu1 = F.relu6(self.mu1(com2))
        mu = self.mu(mu1)
        
        sigma1 = F.relu6(self.sigma1(com2))
        sigma = F.softplus(self.sigma(sigma1)) 

        c1 = F.relu6(self.c1(x))
        c2 = F.relu6(self.c2(c1))
        c3 = F.relu6(self.c3(c2))
        values = self.v(c3)
        
        return mu, sigma, values

    def choose_action(self, s, t = 0):
        self.training = False
        mu, sigma, _ = self.forward(s)
        if t % 20 == 0:
            print(mu[0], sigma[0])
        if is_gpu_available:
            mu, sigma = mu.cuda(), sigma.cuda()
        
        m = self.distribution(mu.view(self.a_dim, ).data, sigma.view(self.a_dim, ).data)
        y = m.sample().cpu().numpy()
        return y

    def loss_func(self, s, a, v_t):
        self.train()
        mu, sigma, values = self.forward(s)
        
        if is_gpu_available:
            mu, sigma, values = mu.cuda(), sigma.cuda(), values.cuda()
        
        td = v_t - values
        c_loss = td.pow(2)

        m = self.distribution(mu, sigma)
        log_prob = m.log_prob(a)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale)  # exploration
        exp_v = log_prob * td.detach() + 0.005 * entropy
        a_loss = -exp_v
        total_loss = (a_loss + c_loss).mean()
        return total_loss              
        
class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, agent_port, monitor_port, name):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.env = Environment(agent_port=agent_port, monitor_port=monitor_port)
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)
        
        if is_gpu_available:
            self.lnet = self.lnet.cuda()
            

    def run(self):
        # For each episode
        while self.g_ep.value < MAX_EP:
            
            # Save intermediate model
            if self.g_ep.value % 20 == 19:
                torch.save(self.gnet, modelName)
            if self.g_ep.value % 1000 == 999:
                torch.save(self.gnet, "net_" + str(self.g_ep.value//1000) + '.pt')

            # Reset rewards, state
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            s = self.env.reset()

            # Simulate this episode
            for t in range(MAX_EP_STEP):
                
                # Sample action from model and step into environment
                if is_gpu_available:
                    a = self.lnet.choose_action(torch.from_numpy(s).float().cuda(), t)
                else:
                    a = self.lnet.choose_action(v_wrap(s[:]), t)

                s_, r, done, _ = self.env.step(a)
                
                # Collect Rewards, States, Actions for later
                if s is not None:
                    ep_r += r
                    buffer_s.append(s)
                    buffer_a.append(a)
                    buffer_r.append(r)
                else:
                    break

                s = s_

                # Early Termination
                if done or t == MAX_EP_STEP-1:  
                    # Update Global And Local Neural Nets After This Episode
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA, is_gpu_available)
                    print("(Total_Reward)",ep_r)                    
                    
                    # Record the cumulative reward and update average
                    buffer_s, buffer_a, buffer_r = [], [], []
                    record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                    break

        self.res_queue.put(None)
 
    def cleanup(self):
        self.env.cleanup()

def test():
    try:
        gnet = torch.load(modelName)
        e = Environment()
        s = e.reset()
        for t in range(MAX_EP_STEP):
            s, _, done, _ = e.step(gnet.choose_action(v_wrap(s[:])))  
            if done: 
                break  
        e.cleanup()
    
    except(KeyboardInterrupt, SystemExit): 
        e.cleanup()

if __name__ == "__main__":
    try:
        if is_gpu_available:
            torch.multiprocessing.set_start_method("spawn")
            torch.cuda.init()
            torch.cuda.device(0)

        if testModel:
            test()
            sys.exit()

        # Load Intermediate model
        if loadModel:
            gnet = torch.load(modelName)
        else:
            gnet = Net(N_S, N_A)
        
        # Model Pytorch Settings
        gnet.share_memory() 
        opt = SharedAdam(gnet.parameters(), lr=learning_rate) 
        global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

        # Parallel Training
        agent_port = 3100
        monitor_port = 3200
        workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, agent_port + i, monitor_port + i, i) for i in range(0,NUM_THREADS)]
        
        # Train
        [w.start() for w in workers]
    
        print("HH1")
        # Get moving averages from training
        res = []
        while True:
            r = res_queue.get()
            if r is not None:
                res.append(r)
            else:
                break
            [w.join() for w in workers]
            
        # Draw the learning Curve (Moving average)
        import matplotlib.pyplot as plt
        plt.plot(res)
        plt.ylabel('Moving average ep reward')
        plt.xlabel('Step')
        plt.savefig("lc.png")
        
        print("HH2")
        #Cleanup
        for w in workers:
            w.cleanup()
        sys.exit()

    except(KeyboardInterrupt, SystemExit):
        sys.exit()
        
