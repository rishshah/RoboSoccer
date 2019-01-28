import sys
sys.path.append('../')
sys.path.append('../environment')

import time
import torch
import torch.nn as nn
from utils import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
from environment.environment import Environment
import math, os, gym

NUM_THREADS = 1
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)

UPDATE_GLOBAL_ITER = 5
GAMMA = 1
MAX_EP = 500
MAX_EP_STEP = 600
TEST = False
env = Environment()
N_S = env.state_dim
N_A = env.action_dim
Z = 60
SPAN = 0.1 # in radians per sec
DELTA = 0.0001 # minimum sigma

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
        sigma = 0.1 * F.softplus(self.sigma(b3)) + DELTA      # avoid 0

        
        c1 = F.relu(self.c1(x))
        c2 = F.relu(self.c2(c1))
        c3 = F.relu(self.c3(c2))
        values = self.v(c3)
        
        return mu, sigma, values

    def choose_action(self, s):
        self.training = False
        mu, sigma, _ = self.forward(s)
        # print(mu)
        # print(sigma)
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
        self.lnet = Net(N_S, N_A)           # local network
        self.env = Environment(agent_port=agent_port, monitor_port=monitor_port)

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            if self.g_ep.value % 20 == 0:
                torch.save(self.gnet, 'int_net_'+str(self.g_ep.value)+'.pt')

            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            s = self.env.reset()
            for t in range(MAX_EP_STEP):
                a = self.lnet.choose_action(v_wrap(s[:]))
                s_, r, done, _ = self.env.step(a)
                
                if s_ is not None:
                    ep_r += r
                    buffer_s.append(s)
                    buffer_a.append(a)
                    buffer_r.append(r)
                else:
                    continue

                if t >= MAX_EP_STEP - 1:
                    done = True
                
                if done:  # update global and assign to local net
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                    # time.sleep(0.1)
                    break

                s = s_
                total_step += 1

        self.res_queue.put(None)

def test(env, gnet):
    # for x in gnet.a1.parameters():
    #     print(x.data)
    for x in gnet.a1.parameters():
        print(x.data)
    # s = env.reset()
    # for t in range(10*MAX_EP_STEP):
    #     a = gnet.choose_action(v_wrap(s[:]))
    #     s, r, done, _ = env.step(env.clip_action(a, s))

if __name__ == "__main__":
    try:
        if TEST:
            test(Environment(), torch.load('int_net_60.pt'))
            env.cleanup()
            sys.exit()

        gnet = Net(N_S, N_A)        # global network
        gnet.share_memory()         # share the global parameters in multiprocessing
        opt = SharedAdam(gnet.parameters(), lr=0.001)  # global optimizer
        global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

        # parallel training
        agent_ports = [3100, 3101, 3102, 3103]
        monitor_ports = [3200, 3201, 3202, 3203]
        workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, agent_ports[i], monitor_ports[i], i) for i in range(0,NUM_THREADS)]
        [w.start() for w in workers]
        res = []                    # record episode reward to plot
        while True:
            r = res_queue.get()
            if r is not None:
                res.append(r)
            else:
                break
            [w.join() for w in workers]
            
        torch.save(gnet, 'gloal_net.pt')
        
        import matplotlib.pyplot as plt
        plt.plot(res)
        plt.ylabel('Moving average ep reward')
        plt.xlabel('Step')
        plt.show()
        
        env.cleanup()
        sys.exit()

    except(KeyboardInterrupt, SystemExit):
        env.cleanup()
        sys.exit()
        