from maze_env import Maze
from DDQN_RL_brain import DoubleDQN
from DDQN_RL_brain import DDQNWithPresetReplay
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


to_plot_steps = []
to_plot_rewards = []
labels = []

max_episode = 800



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
    env = Maze()

    with tf.variable_scope('supervised'):
        RL = DoubleDQN(env.n_actions, env.n_features,
                          learning_rate=0.01,
                          reward_decay=0.9,
                          e_greedy=0.9,
                          replace_target_iter=200,
                          memory_size=2000,
                          # output_graph=True
                          )

    with open(save_path, 'rb') as f:
        train_data = pickle.load(f)

    X = train_data[:, :env.n_features]
    Y_ = train_data[:, env.n_features].astype(np.uint32)
    Y = np.zeros((Y_.shape[0], env.n_actions))


    # Y[np.arange(Y_.shape[0]), Y_] = 1
    Y += 1. / (env.n_actions + 1)
    Y[np.arange(Y_.shape[0]), Y_] += 1. / (env.n_actions + 1)

    RL.mimic_learn(X, Y)

    labels.append('soft-supervised')

    env.after(100, run_maze)
    env.mainloop()

    print('second start')

    env = Maze()

    with tf.variable_scope('ddqn'):
        RL = DoubleDQN(env.n_actions, env.n_features,
                          learning_rate=0.01,
                          reward_decay=0.9,
                          e_greedy=0.9,
                          replace_target_iter=200,
                          memory_size=2000,
                          # output_graph=True
                          )

    labels.append('ddqn')
    env.after(100, run_maze)
    env.mainloop()



    print('third start')

    env = Maze()

    with tf.variable_scope('strong-supervised'):
        RL = DoubleDQN(env.n_actions, env.n_features,
                          learning_rate=0.01,
                          reward_decay=0.9,
                          e_greedy=0.9,
                          replace_target_iter=200,
                          memory_size=2000,
                          # output_graph=True
                          )

    with open(save_path, 'rb') as f:
        train_data = pickle.load(f)

    X = train_data[:, :env.n_features]
    Y_ = train_data[:, env.n_features].astype(np.uint32)
    Y = np.zeros((Y_.shape[0], env.n_actions))

    Y[np.arange(Y_.shape[0]), Y_] = 1

    RL.mimic_learn(X, Y)

    labels.append('strong-supervised')
    env.after(100, run_maze)
    env.mainloop()

    print('forth start')

    env = Maze()

    with tf.variable_scope('static-memory-preset'):
        RL = DoubleDQN(env.n_actions, env.n_features,
                          learning_rate=0.01,
                          reward_decay=0.9,
                          e_greedy=0.9,
                          replace_target_iter=200,
                          memory_size=2000,
                          # output_graph=True
                          )
    with open(save_path, 'rb') as f:
        train_data = pickle.load(f)

    for memory in train_data:
        RL.store_transition(memory[:env.n_features], memory[env.n_features], memory[env.n_features + 1],
                            memory[-env.n_features:])

    labels.append('static-memory-preset')
    env.after(100, run_maze)
    env.mainloop()





    print('fifth start')
    env = Maze()
    with tf.variable_scope('prior-memory'):
        RL = DDQNWithPresetReplay(env.n_actions, env.n_features,
                          learning_rate=0.01,
                          reward_decay=0.9,
                          e_greedy=0.9,
                          replace_target_iter=200,
                          memory_size=2000,
                          preset_replay_base=0
                          # output_graph=True
                          )
    with open(save_path, 'rb') as f:
        train_data = pickle.load(f)
    RL.preset_memory(train_data)
    labels.append('prior-memory')
    env.after(100, run_maze)
    env.mainloop()

    # print('forth start :v.2')

    # env = Maze()

    # with tf.variable_scope('preset-memory-full'):
    #     RL = DoubleDQN(env.n_actions, env.n_features,
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
    # while True:
    #     index = RL.memory_counter if hasattr(RL, 'memory_counter') else 0
    #     if index == RL.memory_size:
    #         break
    #     index = index % len(train_data)
    #     RL.store_transition(train_data[index, :env.n_features], train_data[index, env.n_features],
    #                         train_data[index, env.n_features + 1], train_data[index, -env.n_features:])
    #
    # labels.append('preset-memory-v2')
    # env.after(100, run_maze)
    # env.mainloop()

    to_plots = [
        ['soft-supervised', 'ddqn'],
        ['strong-supervised', 'ddqn'],
        ['strong-supervised', 'soft-supervised', 'ddqn'],
        ['ddqn', 'static-memory-preset'],
        ['ddqn', 'prior-memory'],
        ['ddqn', 'static-memory-preset', 'prior-memory'],
        ['strong-supervised', 'soft-supervised', 'ddqn', 'static-memory-preset', 'prior-memory']
    ]

    def plotttt(lbs):
        plt.figure()
        for lb in lbs:
            idx = labels.index(lb)
            plt.plot(to_plot_steps[idx], to_plot_rewards[idx], label=lb)
        plt.xlabel('step')
        plt.ylabel('average reward')
        plt.legend()
        plt.grid()
        plt.show()

    for to_plot in to_plots:
        plotttt(to_plot)

    # for i, (t, r) in enumerate(zip(to_plot_steps, to_plot_rewards)):
    #     plt.plot(t, r, label=labels[i])
    #     print('total step {}, reward {}'.format(t.shape[0], r[-1]))
    # plt.xlabel('step')
    # plt.ylabel('average reward')
    # plt.legend()
    # plt.grid()
    # plt.show()




