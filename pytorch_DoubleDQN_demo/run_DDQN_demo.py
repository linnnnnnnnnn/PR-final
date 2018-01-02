from Double_DQN import DDQN_brain
from Double_DQN import DQN_brain
from Double_DQN import DDQN_brain_with_experience

import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'state_'))

from maze_env import Maze

def train(qn):
    step = 0
    steps = []
    rewards = []
    episodes = []
    to_plot_steps = []
    to_plot_rewards = []

    for i_episode in range(500):
        s = env.reset()
        while True:
            # env.render()
            a = qn.choose_action(s)
            s_, r, done = env.step(a)

            qn.store_transition(s, a, r, s_)

            if qn.memory_ready():
                qn.learn()

            rewards.append(r)
            if done:
                # print('episode ', i_episode, ' finished')
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

def test():
    his_ddqn = train(DDQN_brain(env.n_actions, env.n_features))

    ddqn_test = DDQN_brain(env.n_actions, env.n_features)
    memory = pickle.load(open("../models/mimic.pickle", "r"))
    memory = [Transition(memory[i, 0:2], int(memory[i, 2].astype(np.int32)), memory[i, 3], memory[i, 4:]) for i in range(memory.shape[0])]
    ddqn_test.init_transition(memory)
    his_ddqn_test = train(ddqn_test)

    ddqn_test2 = DDQN_brain_with_experience(env.n_actions, env.n_features)
    ddqn_test2.init_transition(memory)
    his_ddqn_test2 = train(ddqn_test2)

    plt.plot(his_ddqn[0, :], his_ddqn[1, :], c='b', label='old')
    plt.plot(his_ddqn_test[0, :], his_ddqn_test[1, :], c='r', label='ddqn_with_initial_experience')
    plt.plot(his_ddqn_test2[0, :], his_ddqn_test2[1, :], c='g', label='ddqn_with_proportional_experience')


    plt.legend(loc='best')
    plt.ylabel('average rewards')
    plt.xlabel('steps')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    env = Maze()
    env.after(100, test)
    env.mainloop()