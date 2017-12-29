import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import random
from collections import namedtuple

class DQN_net(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN_net, self).__init__()
        self.conv1 = nn.Linear(input_size, 50)
        self.conv1.weight.data.normal_(0, 0.1)
        self.conv2 = nn.Linear(50, output_size)
        self.conv2.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        return self.conv2(x)


class DQN_brain(object):
    def __init__(self, action_n, state_n, memory_size=2000, lr=0.01, gama=0.9, epsilon=0.9, target_iteration=100):
        self.gama = gama
        self.epsilon = epsilon
        self.action_n = action_n
        self.state_n = state_n

        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'state_'))
        self.memory_position = 0
        self.memory_size = memory_size
        self.memory = []
        self.learn_step_counter = 0
        self.target_replace_iter = target_iteration

        self.Q_target = DQN_net(state_n, action_n)
        self.Q_eval = DQN_net(state_n, action_n)
        self.optimizer = torch.optim.Adam(self.Q_eval.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()

        self.cost_his = []

    def choose_action(self, x):
        if isinstance(x, list):
            x = np.array(x)
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        if np.random.uniform() < self.epsilon:
            action = self.Q_eval.forward(x)
            action = torch.max(action, 1)[1].data.numpy()[0]
        else:
            action = np.random.randint(0, self.action_n)
        return action

    def store_transition(self, s, a, r, s_):
        if isinstance(s, list):
            s = np.array(s)
        if isinstance(s_, list):
            s_ = np.array(s_)
        if len(self.memory) < self.memory_size:
            self.memory.append(None)
        self.memory[self.memory_position] = self.Transition(s, a, r, s_)
        self.memory_position = (self.memory_position + 1) % self.memory_size

    def learn(self, batch_size=32):
        samples = random.sample(self.memory, batch_size)
        samples_s = Variable(torch.FloatTensor([sample.state for sample in samples]))
        samples_a = Variable(torch.LongTensor([sample.action for sample in samples]))
        samples_r = Variable(torch.FloatTensor([sample.reward for sample in samples]))
        samples_s_ = Variable(torch.FloatTensor([sample.state_ for sample in samples]))

        esti_value, real_value = self.brain_core(samples_s, samples_a, samples_r, samples_s_)

        loss = self.loss_func(esti_value, real_value)
        self.cost_his.append(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict())

    def brain_core(self, samples_s, samples_a, samples_r, samples_s_):
        esti_value = self.Q_eval(samples_s).gather(1, samples_a.view(-1, 1))
        next_value = self.Q_target(samples_s_).detach()
        real_value = samples_r + self.gama * next_value.max(1)[0]

        return esti_value, real_value

    def memory_ready(self):
        return len(self.memory) == self.memory_size
    def get_memory_size(self):
        return len(self.memory)


class DDQN_brain(DQN_brain):
    def __init__(self, action_n, state_n, memory_size=2000, lr=0.01, gama=0.9, epsilon=0.9, target_iteration=100):
        super(DDQN_brain, self).__init__(action_n, state_n, memory_size, lr, gama, epsilon, target_iteration)

    def brain_core(self, samples_s, samples_a, samples_r, samples_s_):
        esti_value = self.Q_eval(samples_s).gather(1, samples_a.view(-1, 1))

        next_action = self.Q_eval(samples_s_).max(1)[1]
        next_value = self.Q_target(samples_s_).detach().gather(1, next_action.view(-1, 1))

        real_value = samples_r.view(-1, 1) + self.gama * next_value

        return esti_value, real_value

    def init_transition(self, memory):
        if len(memory) > self.memory_size:
            self.memory = memory
        else:
            self.memory = memory[-1*self.memory_size:]
        self.memory_size = len(memory) % self.memory_size

