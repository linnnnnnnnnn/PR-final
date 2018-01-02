from maze_env import Maze
from RL_brain import DeepQNetwork
from RL_brain import DeepQNetworkWithPresetReplay
from DQN_modified import DeepQNetwork as MDeepQNetwork
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


to_plot_steps = []
to_plot_rewards = []

max_episode = 100


def run_maze():
    rewards = []
    total_reward = 0
    step = 0
    for episode in range(max_episode):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            # env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            rewards.append(reward)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()
                pass

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                print('episode {}'.format(episode))
                break
            step += 1

    # end of game
    print('game over')
    rewards = np.array(rewards, dtype=np.float32)
    steps = np.arange(rewards.shape[0]) + 1
    rewards = np.cumsum(rewards)
    r = rewards * 1. / steps
    to_plot_steps.append(steps)
    to_plot_rewards.append(r)
    env.destroy()


save_path = 'models/mimic.pickle'

if __name__ == "__main__":
    # maze game
    # env = Maze()
    #
    # with tf.variable_scope('supervised'):
    #     RL = DeepQNetwork(env.n_actions, env.n_features,
    #                       learning_rate=0.01,
    #                       reward_decay=0.9,
    #                       e_greedy=0.9,
    #                       replace_target_iter=200,
    #                       memory_size=2000,
    #                       # output_graph=True
    #                       )
    #
    # with open(save_path, 'rb') as f:
    #     train_data = pickle.load(f)
    #
    # X = train_data[:, :env.n_features]
    # Y_ = train_data[:, env.n_features].astype(np.uint32)
    # Y = np.zeros((Y_.shape[0], env.n_actions))
    #
    #
    # # Y[np.arange(Y_.shape[0]), Y_] = 1
    # Y += 1. / (env.n_actions + 1)
    # Y[np.arange(Y_.shape[0]), Y_] += 1. / (env.n_actions + 1)
    #
    # RL.mimic_learn(X, Y)
    #
    # env.after(100, run_maze)
    # env.mainloop()
    #
    # print('second start')
    #
    # env = Maze()
    #
    # with tf.variable_scope('nature'):
    #     RL = DeepQNetwork(env.n_actions, env.n_features,
    #                       learning_rate=0.01,
    #                       reward_decay=0.9,
    #                       e_greedy=0.9,
    #                       replace_target_iter=200,
    #                       memory_size=2000,
    #                       # output_graph=True
    #                       )
    #
    # env.after(100, run_maze)
    # env.mainloop()
    #
    # print('third start')
    #
    # env = Maze()
    #
    # with tf.variable_scope('strong-supervised'):
    #     RL = DeepQNetwork(env.n_actions, env.n_features,
    #                       learning_rate=0.01,
    #                       reward_decay=0.9,
    #                       e_greedy=0.9,
    #                       replace_target_iter=200,
    #                       memory_size=2000,
    #                       # output_graph=True
    #                       )
    #
    # with open(save_path, 'rb') as f:
    #     train_data = pickle.load(f)
    #
    # X = train_data[:, :env.n_features]
    # Y_ = train_data[:, env.n_features].astype(np.uint32)
    # Y = np.zeros((Y_.shape[0], env.n_actions))
    #
    # Y[np.arange(Y_.shape[0]), Y_] = 1
    #
    # RL.mimic_learn(X, Y)
    #
    # env.after(100, run_maze)
    # env.mainloop()
    #
    # print('forth start')
    #
    # env = Maze()
    #
    # with tf.variable_scope('init-memory'):
    #     RL = DeepQNetwork(env.n_actions, env.n_features,
    #                       learning_rate=0.01,
    #                       reward_decay=0.9,
    #                       e_greedy=0.9,
    #                       replace_target_iter=200,
    #                       memory_size=2000,
    #                       # output_graph=True
    #                       )
    # with open(save_path, 'rb') as f:
    #     train_data = pickle.load(f)
    #
    # for memory in train_data:
    #     RL.store_transition(memory[:env.n_features], memory[env.n_features], memory[env.n_features+1], memory[-env.n_features:])
    #
    # env.after(100, run_maze)
    # env.mainloop()

    print('fifth start')

    env = Maze()

    with tf.variable_scope('initial-memory-proportional-replay'):
        RL = DeepQNetworkWithPresetReplay(env.n_actions, env.n_features,
                          learning_rate=0.01,
                          reward_decay=0.9,
                          e_greedy=0.9,
                          replace_target_iter=200,
                          memory_size=2000,
                          # output_graph=True
                          )
    with open(save_path, 'rb') as f:
        train_data = pickle.load(f)

    RL.preset_memory(train_data)
    env.after(100, run_maze)
    env.mainloop()




    for i, (t, r) in enumerate(zip(to_plot_steps, to_plot_rewards)):
        if i == 0:
            label = 'with soft supervised'
        elif i == 1:
            label = 'without supervised'
        else:
            label = 'with strong supervised'
        plt.plot(t, r, label=label)
        print('total step {}, reward {}'.format(t.shape[0], r[-1]))
    plt.xlabel('step')
    plt.ylabel('average reward')
    plt.legend()
    plt.grid()
    plt.show()




