import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import math, os

sys.path.append('../')
sys.path.append('../Environment')

from utils import v_wrap, set_init, push_and_pull, record
from shared_adam import SharedAdam
from Environment.environment import Environment
import copy 

# Global Variables and HyperParameters
NUM_THREADS = 1
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)

GAMMA = 1
MAX_EP = 1000 # Max episodes
MAX_EP_STEP = 600 # Max episode step

env = Environment()
N_S = env.state_dim
N_A = env.action_dim

Z = 70 # number of hidden nodes in each layer
SPAN = 0.5 # in radians per sec
DELTA = 0.0001 # minimum sigma

modelName = 'int_net.pt'
loadModel = False
learning_rate = 0.001

class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        
        self.a1 = nn.Linear(s_dim, Z)
        self.a2 = nn.Linear(Z, Z)
        self.a3 = nn.Linear(Z, Z)
        
        self.b1 = nn.Linear(s_dim, Z)
        self.b2 = nn.Linear(Z, Z)
        self.b3 = nn.Linear(Z, Z)
        
        self.mu = nn.Linear(Z, a_dim)
        self.sigma = nn.Linear(Z, a_dim)

        self.c1 = nn.Linear(s_dim, Z)
        self.c2 = nn.Linear(Z, Z)
        self.c3 = nn.Linear(Z, Z)
        self.v = nn.Linear(Z, 1)
        
        set_init([self.a1, self.a2, self.a3, self.b1, self.b2, self.b3, self.mu, self.sigma, self.c1, self.c2, self.c3, self.v])
        self.distribution = torch.distributions.Normal

    def forward(self, x):
        a1 = F.relu(self.a1(x))
        a2 = F.relu(self.a2(a1))
        a3 = F.relu(self.a3(a2))
        mu = SPAN * torch.tanh(self.mu(a3))

        b1 = F.relu(self.b1(x))
        b2 = F.relu(self.b2(b1))
        b3 = F.relu(self.b3(b2))
        sigma = F.softplus(self.sigma(b3)) + DELTA # TODO Is delta necessary 

        c1 = F.relu(self.c1(x))
        c2 = F.relu(self.c2(c1))
        c3 = F.relu(self.c3(c2))
        values = self.v(c3)
        
        return mu, sigma, values

    def choose_action(self, s):
        self.training = False
        mu, sigma, val = self.forward(s)
        # print("(mu)", mu)
        # print("(sigma)", sigma)
        # print("(val)", val)
        m = self.distribution(mu.view(self.a_dim, ).data, sigma.view(self.a_dim, ).data)
        y = m.sample().numpy()
        return y

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
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, agent_port, monitor_port, name):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = copy.deepcopy(self.gnet) #TODO self.lnet = Net(N_S, N_A)
        self.env = Environment(agent_port=agent_port, monitor_port=monitor_port)

    def run(self):
        # For each episode
        while self.g_ep.value < MAX_EP:
            
            # Save intermediate model
            if self.g_ep.value % 20 == 0:
                torch.save(self.gnet, modelName)

            # Reset rewards, state
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            s = self.env.reset()

            # Simulate this episode
            for t in range(MAX_EP_STEP):
                
                # Sample action from model and step into environment
                a = self.lnet.choose_action(v_wrap(s[:]))
                s_, r, done, _ = self.env.step(a)
                
                # Collect Rewards, States, Actions for later
                if s_ is not None:
                    ep_r += r
                    buffer_s.append(s)
                    buffer_a.append(a)
                    buffer_r.append(r)
                else:
                    done = False
                    break
                
                # Update new state
                s = s_
                
                # Early Termination
                if done:
                    break

                # Terminate after MAX_EP_STEP
                if t == MAX_EP_STEP:
                    done = True

            if done:  
                print("(Total_Reward)",ep_r)
                print("MIN", self.env.min_acc_x, self.env.min_acc_y)
                print("MAX", self.env.max_acc_x, self.env.max_acc_y)
                
                # Update Global And Local Neural Nets After This Episode
                push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                
                # Record the cumulative reward and update average
                buffer_s, buffer_a, buffer_r = [], [], []
                record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)

        self.res_queue.put(None)

def test():
    gnet = torch.load(modelName)
    s = env.reset()
    for t in range(MAX_EP_STEP):
        a = gnet.choose_action(v_wrap(s[:]))
        s, r, done, _ = env.step(env.clip_action(a, s))    

if __name__ == "__main__":
    try:
        # test()
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
        agent_ports = [3100, 3101, 3102, 3103]
        monitor_ports = [3200, 3201, 3202, 3203]
        workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, agent_ports[i], monitor_ports[i], i) for i in range(0,NUM_THREADS)]
        
        # Train
        [w.start() for w in workers]
        
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
        plt.show()
        
        #Cleanup
        env.cleanup()
        sys.exit()

    except(KeyboardInterrupt, SystemExit):
        env.cleanup()
        sys.exit()
        