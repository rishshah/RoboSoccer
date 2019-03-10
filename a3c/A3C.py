import sys
import torch
import torch.nn as nn
from utils import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
import gym
import math, os
import matplotlib.pyplot as plt

sys.path.append('../')
sys.path.append('../Environment')
from Environment.environment import Environment

os.environ["OMP_NUM_THREADS"] = "1"

# Training Hyperparameters
UPDATE_GLOBAL_ITER = 50
GAMMA = 0.99
MAX_EP = 10000
MAX_EP_STEP = 100
LEARNING_RATE = 0.0003
NUM_WORKERS = mp.cpu_count()

# Model IO Parameters
MODEL_NAME = "stand"
LOAD_MODEL = False
TEST_MODEL = False

# Neural Network Architecture Variables
ENV_DUMMY = Environment()
N_S, N_A = ENV_DUMMY.state_dim, ENV_DUMMY.action_dim
Z1 = 100
Z2 = 80
MU_SPAN = 1

# Gpu use flag
# is_gpu_available = torch.cuda.is_available()

class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a1 = nn.Linear(s_dim, Z1)
        self.a2 = nn.Linear(Z1, Z2)
        self.mu = nn.Linear(Z2, a_dim)
        self.sigma = nn.Linear(Z2, a_dim)
        self.c1 = nn.Linear(s_dim, Z1)
        self.c2 = nn.Linear(Z1, Z2)
        self.v = nn.Linear(Z2, 1)
        set_init([self.a1, self.a2, self.mu, self.sigma, self.c1, self.c2, self.v])
        self.distribution = torch.distributions.Normal

    def forward(self, x):
        a1 = F.relu6(self.a1(x))
        a2 = F.relu6(self.a2(a1))
        # mu = MU_SPAN * torch.tanh(self.mu(a2))
        mu = self.mu(a2)
        sigma = F.softplus(self.sigma(a2))
        c1 = F.relu6(self.c1(x))
        c2 = F.relu6(self.c2(c1))
        values = self.v(c2)
        return mu, sigma, values

    def choose_action(self, s, t=0):
        self.training = False
        mu, sigma, _ = self.forward(s)
        if(t % 20 == 0):
            print(mu[0][0], sigma[0][0])
        m = self.distribution(mu.view(self.a_dim, ).data, sigma.view(self.a_dim, ).data)
        # if t == -1:
        #     return mu.detach().numpy()
        return m.sample().numpy()

    def loss_func(self, s, a, v_t):
        self.train()
        mu, sigma, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        # if is_gpu_available:
        #     mu, sigma, values = mu.cuda(), sigma.cuda(), values.cuda()

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
        self.env = Environment(agent_port=agent_port, monitor_port=monitor_port)
        self.lnet = Net(N_S, N_A)

        # if is_gpu_available:
        #     self.lnet = self.lnet.cuda()


    def run(self):
        try:
            while self.g_ep.value < MAX_EP:
                s = self.env.reset()
                buffer_s, buffer_a, buffer_r = [], [], []
                ep_r = 0.

                if self.g_ep.value % 20 == 19:
                    torch.save(self.gnet, MODEL_NAME + ".pt")

                for t in range(MAX_EP_STEP):
                    a = self.lnet.choose_action(v_wrap(s[None, :]), t)
                    # if is_gpu_available:
                    #     a = self.lnet.choose_action(torch.from_numpy(s).float().cuda(), t)
                    # else:
                    #     a = self.lnet.choose_action(v_wrap(s[:]), t)
                    s_, r, done, _ = self.env.step(a)

                    if t == MAX_EP_STEP - 1:
                        done = True
                    
                    if s is not None:
                        ep_r += r
                        buffer_a.append(a)
                        buffer_s.append(s)
                        buffer_r.append(r)

                    if done:  # Sync Global and Local Nets
                        push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                    s = s_
            
            self.res_queue.put(None)
            self.env.cleanup()
        
        except(KeyboardInterrupt):
            self.res_queue.put(None)
            self.env.cleanup()

def test():
    try:
        gnet = torch.load(MODEL_NAME + ".pt")
        s = ENV_DUMMY.reset()
        for t in range(MAX_EP_STEP):
            s, _, done, _ = ENV_DUMMY.step(gnet.choose_action(v_wrap(s[:]), -1))  
            if done: 
                break  
        ENV_DUMMY.cleanup()
    
    except(KeyboardInterrupt): 
        ENV_DUMMY.cleanup()

if __name__ == "__main__":
    # if is_gpu_available:
    #     torch.multiprocessing.set_start_method("spawn")
    #     torch.cuda.init()
    #     torch.cuda.device(0)

    # Initialize Global Net and Optimizer     
    if TEST_MODEL:
        test()
        sys.exit()

    if LOAD_MODEL:
        gnet = torch.load(MODEL_NAME + ".pt")
    else:
        gnet = Net(N_S, N_A)

    gnet.share_memory()        
    opt = SharedAdam(gnet.parameters(), lr=LEARNING_RATE)
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # Parallel training
    agent_port = 3100
    monitor_port = 3200
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i, agent_port + i, monitor_port + i) for i in range(NUM_WORKERS)]
    [w.start() for w in workers]
    
    # Plot moving average of rewards
    res = []
    while True:
        try:
            r = res_queue.get()
            if r is not None:
                res.append(r)
            else:
                break
        except(KeyboardInterrupt):
            break
    [w.join() for w in workers]
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Episodes')
    plt.savefig(MODEL_NAME + "_lc.png")


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
# self.c2.register_backward_hook(printgradnorm)