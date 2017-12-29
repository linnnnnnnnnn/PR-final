from Double_DQN import DDQN_brain
from Double_DQN import DQN_brain

import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'state_'))

from maze_env import Maze

def train(qn):
    step = 0
    steps = []
    episodes = []
    for i_episode in range(20):
        s = env.reset()
        while True:
            # env.render()
            a = qn.choose_action(s)
            s_, r, done = env.step(a)
            if done: r = 10
            qn.store_transition(s, a, r, s_)

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
    return np.vstack((episodes, steps))

def test():
    his_ddqn = train(DQN_brain(env.n_actions, env.n_features))

    ddqn_test = DDQN_brain(env.n_actions, env.n_features)
    memory = pickle.load(open("../models/mimic.pickle", "r"))
    memory = [Transition(memory[i, 0:2], int(memory[i, 2].astype(np.int32)), memory[i, 3], memory[i, 4:]) for i in range(memory.shape[0])]
    ddqn_test.init_transition(memory)
    his_ddqn_test = train(ddqn_test)


    # compare based on first success
    plt.plot(his_ddqn[0, :], his_ddqn[1, :] - his_ddqn[1, 0], c='b', label='DQN')
    plt.plot(his_ddqn_test[0, :], his_ddqn_test[1, :] - his_ddqn_test[1, 0], c='r', label='DDQN')
    plt.legend(loc='best')
    plt.ylabel('total training time')
    plt.xlabel('episode')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    env = Maze()
    env.after(100, test)
    env.mainloop()
