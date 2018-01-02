import gym
from Double_DQN import DDQN_brain
from Double_DQN import DQN_brain
from Double_DQN import DDQN_brain_with_experience

import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'state_'))

env = gym.make('MountainCar-v0')
env = env.unwrapped
N_ACTIONS = 3
N_STATES = 2

def train(qn):
    step = 0
    steps = []
    episodes = []
    rewards = []
    to_plot_steps = []
    to_plot_rewards = []

    for i_episode in range(200):
        s = env.reset()
        while True:
            # env.render()
            a = qn.choose_action(s)
            s_, r, done, info = env.step(a)
            # if done: r = 10
            qn.store_transition(s, a, r, s_)

            rewards.append(r)
            if qn.memory_ready():
                qn.learn()

            if done:
                print('episode ', i_episode, ' finished')
                steps.append(step)
                episodes.append(i_episode)
                break

            step += 1
            s = s_
    # env.render(close=True)
    rewards = np.array(rewards, dtype=np.float32)
    steps = np.arange(rewards.shape[0]) + 1
    rewards = np.cumsum(rewards)
    r = rewards * 1. / steps
    to_plot_steps.append(steps)
    to_plot_rewards.append(r)
    print('finished')
    return np.vstack((to_plot_steps, to_plot_rewards))


# test
his_ddqn = train(DDQN_brain(N_ACTIONS, N_STATES))

ddqn_test = DDQN_brain(N_ACTIONS, N_STATES)
memory = pickle.load(open("../models/mimic_car.pickle", "r"))
memory = [Transition(memory[i, 0:2], int(memory[i, 2].astype(np.int32)), memory[i, 3], memory[i, 4:]) for i in range(memory.shape[0])]
ddqn_test.init_transition(memory)
his_ddqn_test = train(ddqn_test)

# ddqn_test2 = DDQN_brain_with_experience(N_ACTIONS, N_STATES)
# ddqn_test2.init_transition(memory)
# his_ddqn_test2 = train(ddqn_test2)

# plot
plt.plot(his_ddqn[0, :], his_ddqn[1, :], c='b', label='old')
plt.plot(his_ddqn_test[0, :], his_ddqn_test[1, :], c='r', label='ddqn_with_initial_experience')
# plt.plot(his_ddqn_test2[0, :], his_ddqn_test2[1, :], c='g', label='ddqn_with_proportional_experience')

plt.legend(loc='best')
plt.ylabel('average rewards')
plt.xlabel('steps')
plt.grid()
plt.show()
