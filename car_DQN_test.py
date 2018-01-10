"""
Deep Q network,

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

# flag = 'train'
flag = 'p'

tmp_file_path = 'tmp/car.pickle'
save_path = 'models/mimic_car.pickle'

max_episode = 40
n_action = 3
n_feature = 2

if flag == 'train':
    import gym
    from DQN_RL_brain import DeepQNetwork
    from DQN_RL_brain import DeepQNetworkWithPresetReplay
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    import tensorflow as tf

    np.random.seed(1)
    tf.set_random_seed(1)

    env = gym.make('MountainCar-v0')
    env = env.unwrapped

    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.high)
    print(env.observation_space.low)

    to_plot_train_steps = []
    labels = []


    def train():
        total_steps = 0
        ep_steps = []
        for i_episode in range(max_episode):
            observation = env.reset()
            ep_r = 0
            while True:
                # env.render()

                action = RL.choose_action(observation)

                observation_, reward, done, info = env.step(action)

                position, velocity = observation_

                # the higher the better
                reward = abs(position - (-0.5))  # r in [0, 1]

                RL.store_transition(observation, action, reward, observation_)

                if total_steps > 1000:
                    RL.learn()

                ep_r += reward
                if done:
                    ep_steps.append(total_steps)
                    print('episode {}'.format(i_episode))
                    break

                observation = observation_
                total_steps += 1

        to_plot_train_steps.append(ep_steps)


    with tf.variable_scope('soft-supervised'):
        RL = DeepQNetwork(n_actions=3, n_features=2, learning_rate=0.001, e_greedy=0.9,
                          replace_target_iter=300, memory_size=3000,
                          e_greedy_increment=0.0002, )

        with open(save_path, 'rb') as f:
            train_data = pickle.load(f)

        X = train_data[:, :n_feature]
        Y_ = train_data[:, n_feature].astype(np.uint32)
        Y = np.zeros((Y_.shape[0], n_action))

        Y += 1. / (n_action + 1)
        Y[np.arange(Y_.shape[0]), Y_] += 1. / (n_action + 1)

        RL.mimic_learn(X, Y)

        labels.append('soft-supervised')
        train()

    with tf.variable_scope('no-supervised'):
        RL = DeepQNetwork(n_actions=3, n_features=2, learning_rate=0.001, e_greedy=0.9,
                          replace_target_iter=300, memory_size=3000,
                          e_greedy_increment=0.0002, )

        labels.append('nature')
        train()


    with tf.variable_scope('strong-supervised'):
        RL = DeepQNetwork(n_actions=3, n_features=2, learning_rate=0.001, e_greedy=0.9,
                          replace_target_iter=300, memory_size=3000,
                          e_greedy_increment=0.0002, )

        with open(save_path, 'rb') as f:
            train_data = pickle.load(f)

        X = train_data[:, :n_feature]
        Y_ = train_data[:, n_feature].astype(np.uint32)
        Y = np.zeros((Y_.shape[0], n_action))

        Y[np.arange(Y_.shape[0]), Y_] = 1

        RL.mimic_learn(X, Y)
        labels.append('strong-supervised')
        train()



    # with tf.variable_scope('memory-preset-v2'):
    #     RL = DeepQNetwork(n_actions=3, n_features=2, learning_rate=0.001, e_greedy=0.9,
    #                       replace_target_iter=300, memory_size=3000,
    #                       e_greedy_increment=0.0002, )
    #     with open(save_path, 'rb') as f:
    #         train_data = pickle.load(f)
    #
    #     while True:
    #         index = RL.memory_counter if hasattr(RL, 'memory_counter') else 0
    #         if index == RL.memory_size:
    #             break
    #         index = index % len(train_data)
    #         RL.store_transition(train_data[index, :n_feature], train_data[index, n_feature],
    #                             train_data[index, n_feature + 1], train_data[index, -n_feature:])
    #
    #     labels.append('memory-preset-v2')
    #     train()


    # with tf.variable_scope('memory-preset-with-proportional-replay-0'):
    #     RL = DeepQNetworkWithPresetReplay(n_actions=3, n_features=2, learning_rate=0.001, e_greedy=0.9,
    #                                       replace_target_iter=300, memory_size=3000,
    #                                       e_greedy_increment=0.0002, preset_replay_base=0)
    #     with open(save_path, 'rb') as f:
    #         train_data = pickle.load(f)
    #     RL.preset_memory(train_data)
    #
    #     labels.append('memory-preset-with-proportional-replay-0')
    #     train()
    #
    # with tf.variable_scope('memory-preset-with-proportional-replay-200'):
    #     RL = DeepQNetworkWithPresetReplay(n_actions=3, n_features=2, learning_rate=0.001, e_greedy=0.9,
    #                                       replace_target_iter=300, memory_size=3000,
    #                                       e_greedy_increment=0.0002)
    #     with open(save_path, 'rb') as f:
    #         train_data = pickle.load(f)
    #     RL.preset_memory(train_data)
    #
    #     labels.append('memory-preset-with-proportional-replay-200')
    #     train()
    #
    with tf.variable_scope('prior-memory'):
        RL = DeepQNetworkWithPresetReplay(n_actions=3, n_features=2, learning_rate=0.001, e_greedy=0.9,
                                          replace_target_iter=300, memory_size=3000,
                                          e_greedy_increment=0.0002)
        with open(save_path, 'rb') as f:
            train_data = pickle.load(f)
        RL.preset_memory(train_data)

        labels.append('prior-memory')
        train()

    with tf.variable_scope('static-memory-preset'):
        RL = DeepQNetwork(n_actions=3, n_features=2, learning_rate=0.001, e_greedy=0.9,
                          replace_target_iter=300, memory_size=3000,
                          e_greedy_increment=0.0002, )
        with open(save_path, 'rb') as f:
            train_data = pickle.load(f)

        for memory in train_data:
            RL.store_transition(memory[:n_feature], memory[n_feature], memory[n_feature + 1],
                                memory[-n_feature:])
        labels.append('static-memory-preset')
        train()



    with open(tmp_file_path, 'wb') as f:
        pickle.dump((to_plot_train_steps, labels), f)

else:
    import matplotlib.pyplot as plt
    import pickle
    import numpy as np
    with open(tmp_file_path, 'rb') as f:
        to_plot_train_steps, labels = pickle.load(f)

    # for i, t in enumerate(to_plot_train_steps):
    #     e = np.arange(max_episode)
    #     plt.plot(e, t, label=labels[i])
    #
    # plt.xlabel('episode')
    # plt.ylabel('total train step')
    # plt.legend()
    # plt.grid()
    # plt.show()

    to_plots = [
        ['soft-supervised', 'nature'],
        ['strong-supervised', 'nature'],
        ['strong-supervised', 'soft-supervised', 'nature'],
        ['nature', 'static-memory-preset'],
        ['nature', 'prior-memory'],
        ['nature', 'static-memory-preset', 'prior-memory'],
        ['strong-supervised', 'soft-supervised', 'nature', 'static-memory-preset', 'prior-memory']
    ]


    def plotttt(lbs):
        plt.figure()
        for lb in lbs:
            idx = labels.index(lb)
            plt.plot(np.arange(max_episode), to_plot_train_steps[idx], label=lb)
        plt.xlabel('episode')
        plt.ylabel('total train step')
        plt.legend()
        plt.grid()
        plt.show()


    for to_plot in to_plots:
        plotttt(to_plot)



