"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.

View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).
"""

import sys
import torch
import torch.nn as nn
from utils import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
import gym
import math, os
sys.path.append('../')
sys.path.append('../Environment')
from Environment.environment import Environment

os.environ["OMP_NUM_THREADS"] = "4"

UPDATE_GLOBAL_ITER = 50
GAMMA = 0.95
MAX_EP = 10000
MAX_EP_STEP = 200

env = Environment()
N_S = env.state_dim
N_A = env.action_dim
# env = gym.make('Pendulum-v0')
# N_S = env.observation_space.shape[0]
# N_A = env.action_space.shape[0]
Z1 = 200
Z2 = 100

class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a1 = nn.Linear(s_dim, Z1)
        self.mu = nn.Linear(Z1, a_dim)
        self.sigma = nn.Linear(Z1, a_dim)
        self.c1 = nn.Linear(s_dim, Z2)
        self.v = nn.Linear(Z2, 1)
        set_init([self.a1, self.mu, self.sigma, self.c1, self.v])
        self.distribution = torch.distributions.Normal

    def forward(self, x):
        a1 = F.relu6(self.a1(x))
        mu = 2 * torch.tanh(self.mu(a1))
        sigma = F.softplus(self.sigma(a1))
        c1 = F.relu6(self.c1(x))
        values = self.v(c1)
        return mu, sigma, values

    def choose_action(self, s, t=0):
        self.training = False
        mu, sigma, _ = self.forward(s)
        if(t % 30 == 0):
            print(mu[0], sigma[0])
        m = self.distribution(mu.view(self.a_dim, ).data, sigma.view(self.a_dim, ).data)
        return m.sample().numpy()

    def loss_func(self, s, a, v_t):
        self.train()
        mu, sigma, values = self.forward(s)
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
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name, agent_port, monitor_port):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)           # local network
        # self.env = gym.make('Pendulum-v0').unwrapped
        self.env = Environment(agent_port=agent_port, monitor_port=monitor_port)

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.

            if self.g_ep.value % 20 == 19:
                torch.save(self.gnet, "int_net.pt")

            for t in range(MAX_EP_STEP):
                a = self.lnet.choose_action(v_wrap(s[None, :]), total_step)
                s_, r, done, _ = self.env.step(a)
                if t == MAX_EP_STEP - 1:
                    done = True
                
                if s is not None:
                    ep_r += r
                    buffer_a.append(a)
                    buffer_s.append(s)
                    buffer_r.append(r)    # normalize

                if done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1

        self.res_queue.put(None)


if __name__ == "__main__":
    gnet = Net(N_S, N_A)        # global network
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=0.0001)  # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    agent_port = 3100
    monitor_port = 3200
    # X = mp.cpu_count()
    X = 1
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i, agent_port + i, monitor_port + i) for i in range(X)]
    [w.start() for w in workers]
    res = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    import matplotlib.pyplot as plt
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.savefig("lc.png")


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.multiprocessing as mp
# import math, os
# import numpy as np
# import gym

# from utils import v_wrap, set_init, push_and_pull, record
# from shared_adam import SharedAdam

# # Global Variables and HyperParameters
# NUM_THREADS = 1
# os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)

# # env = Environment()
# # N_S = env.state_dim
# # N_A = env.action_dim

# env = gym.make('Pendulum-v0')
# N_S = env.observation_space.shape[0]
# N_A = env.action_space.shape[0]

# GAMMA = 0.95
# MAX_EP = 3000
# MAX_EP_STEP = 200
# UPDATE_GLOBAL_ITER = 500

# Z1 = 200
# SPAN = 2

# modelName = 'int_net.pt'
# loadModel = False
# testModel = False
# learning_rate = 0.0002

# is_gpu_available = torch.cuda.is_available()
# # is_gpu_available = False




# def printgradnorm(self, grad_input, grad_output):
#     print('Inside ' + self.__class__.__name__ + ' backward')
#     print('Inside class:' + self.__class__.__name__)
#     print('')
#     # print('grad_input: ', type(grad_input))
#     # print('grad_input[0]: ', type(grad_input[0]))
#     # print('grad_output: ', type(grad_output))
#     # print('grad_output[0]: ', type(grad_output[0]))
#     print('')
#     print('grad_input size:', grad_input[0].size())
#     print('grad_output size:', grad_output[0].size())
#     print('grad_input:', grad_input)
#     print('$$$$\n\n\n')
#     print('grad_input norm:', grad_input[0].norm()) 
#     print('grad_output norm:', grad_output[0].norm())

# class Net(nn.Module):
#     def __init__(self, s_dim, a_dim):
#         super(Net, self).__init__()
#         self.s_dim = s_dim
#         self.a_dim = a_dim

#         self.com1 = nn.Linear(s_dim, Z1)
        
#         self.mu = nn.Linear(Z1, a_dim)
#         self.sigma = nn.Linear(Z1, a_dim)
        

#         self.c1 = nn.Linear(s_dim, 100)
#         self.v = nn.Linear(100, 1)
        
#         # self.c2.register_backward_hook(printgradnorm)

#         set_init([self.com1, self.mu, self.sigma, self.c1, self.v])
#         self.distribution = torch.distributions.Normal

#     def forward(self, x):
#         com1 = F.relu6(self.com1(x))
#         mu = SPAN * torch.tanh(self.mu(com1))        
#         sigma = F.softplus(self.sigma(com1)) 

#         c1 = F.relu6(self.c1(x))
#         values = self.v(c1)
        
#         return mu, sigma, values

#     def choose_action(self, s, t = 0):
#         self.training = False
#         mu, sigma, _ = self.forward(s)
#         if t % 20 == 0:
#             print(mu[0], sigma[0])
#         # if is_gpu_available:
#         #     mu, sigma = mu.cuda(), sigma.cuda()
        
#         m = self.distribution(mu.view(self.a_dim, ).data, sigma.view(self.a_dim, ).data)
#         if t == -1:
#             return mu.cpu().detach().numpy()
#         return m.sample().cpu().numpy()

#     def loss_func(self, s, a, v_t):
#         self.train()
#         mu, sigma, values = self.forward(s)
        
#         if is_gpu_available:
#             mu, sigma, values = mu.cuda(), sigma.cuda(), values.cuda()
        
#         td = v_t - values
#         c_loss = td.pow(2)

#         m = self.distribution(mu, sigma)
#         entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale)  # exploration
#         exp_v = m.log_prob(a) * td.detach() + 0.005 * entropy
#         a_loss = -exp_v
#         total_loss = (a_loss + c_loss).mean()
#         print("ALOSS", a_loss.norm())
#         print("CLOSS", c_loss.norm())
#         return total_loss              
        
# class Worker(mp.Process):
#     def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, agent_port, monitor_port, name):
#         super(Worker, self).__init__()
#         self.name = 'w%i' % name
#         self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
#         # self.env = Environment(agent_port=agent_port, monitor_port=monitor_port)
#         self.env = gym.make('Pendulum-v0').unwrapped
#         self.gnet, self.opt = gnet, opt
#         self.lnet = Net(N_S, N_A)
        
#         if is_gpu_available:
#             self.lnet = self.lnet.cuda()
            

#     def run(self):
#         total_step = 1
#         # For each episode
#         while self.g_ep.value < MAX_EP:
            
#             # Save intermediate model
#             if self.g_ep.value % 20 == 19:
#                 torch.save(self.gnet, modelName)
#             if self.g_ep.value % 1000 == 999:
#                 torch.save(self.gnet, "net_" + str(self.g_ep.value//1000) + '.pt')

#             # Reset rewards, state
#             buffer_s, buffer_a, buffer_r = [], [], []
#             ep_r = 0.
#             s = self.env.reset()

#             # Simulate this episode
#             for t in range(MAX_EP_STEP):
                
#                 # Sample action from model and step into environment
#                 if is_gpu_available:
#                     a = self.lnet.choose_action(torch.from_numpy(s).float().cuda(), t)
#                 else:
#                     a = self.lnet.choose_action(v_wrap(s[:]), t)

#                 s_, r, done, _ = self.env.step(a)
                
#                 # Collect Rewards, States, Actions for later
#                 if s is not None:
#                     ep_r += r
#                     buffer_s.append(s)
#                     buffer_a.append(a)
#                     r = (r+8.1)/8.1
#                     buffer_r.append(r)
#                 else:
#                     total_step = 1
#                     break

#                 s = s_
#                 total_step += 1
#                 # Early Termination
#                 if total_step % UPDATE_GLOBAL_ITER == 0 or done or t == MAX_EP_STEP-1:  
#                     # Update Global And Local Neural Nets After This Episode
#                     push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA, is_gpu_available)
#                     # Record the cumulative reward and update average
#                     buffer_s, buffer_a, buffer_r = [], [], []
#                     record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
#                     break

#         self.res_queue.put(None)
 
#     def cleanup(self):
#         self.env.cleanup()

# def test():
#     try:
#         gnet = torch.load(modelName)
#         e = Environment()
#         s = e.reset()
#         for t in range(MAX_EP_STEP):
#             s, _, done, _ = e.step(gnet.choose_action(v_wrap(s[:]), -1))  
#             if done: 
#                 break  
#         e.cleanup()
    
#     except(KeyboardInterrupt, SystemExit): 
#         e.cleanup()

# if __name__ == "__main__":
#     np.random.seed(0)

#     try:
#         if is_gpu_available:
#             torch.multiprocessing.set_start_method("spawn")
#             torch.cuda.init()
#             torch.cuda.device(0)

#         if testModel:
#             test()
#             sys.exit()

#         # Load Intermediate model
#         if loadModel:
#             gnet = torch.load(modelName)
#         else:
#             gnet = Net(N_S, N_A)
        
#         # Model Pytorch Settings
#         gnet.share_memory() 
#         opt = SharedAdam(gnet.parameters(), lr=learning_rate) 
#         global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

#         # Parallel Training
#         agent_port = 3100
#         monitor_port = 3200
#         workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, agent_port + i, monitor_port + i, i) for i in range(0,NUM_THREADS)]
#         [w.start() for w in workers]
    
#         # Get moving averages from training
#         res = []
#         while True:
#             r = res_queue.get()
#             if r is not None:
#                 res.append(r)
#             else:
#                 break
#             [w.join() for w in workers]
            
#         # Draw the learning Curve (Moving average)
#         import matplotlib.pyplot as plt
#         plt.plot(res)
#         plt.ylabel('Moving average ep reward')
#         plt.xlabel('Step')
#         plt.savefig("lc.png")
        
#         #Cleanup
#         # for w in workers:
#         #     w.cleanup()
#         sys.exit()

#     except(KeyboardInterrupt, SystemExit):
#         sys.exit()
        
