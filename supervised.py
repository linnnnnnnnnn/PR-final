from maze_env import Maze
from RL_brain import DeepQNetwork
from DQN_modified import DeepQNetwork as MDeepQNetwork
import pickle
import numpy as np
import tensorflow as tf


def run_maze():
    total_reward = 0
    step = 0
    for episode in range(300):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            total_reward += reward

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()
                pass

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('game over')
    print('total step {}, total reward {}, average reward {}'.format(step, total_reward, total_reward * 1. / step))
    env.destroy()


save_path = 'models/mimic.pickle'

if __name__ == "__main__":
    # maze game
    env = Maze()

    sess = tf.Session()
    with tf.variable_scope('Double_DQN'):
        RL = DeepQNetwork(env.n_actions, env.n_features,
                          learning_rate=0.01,
                          reward_decay=0.9,
                          e_greedy=0.9,
                          replace_target_iter=200,
                          memory_size=2000,
                          # output_graph=True
                          )

    sess.run(tf.global_variables_initializer())

    with open(save_path, 'rb') as f:
        train_data = pickle.load(f)

    X = train_data[:, :env.n_features]
    Y_ = train_data[:, env.n_features].astype(np.uint32)
    Y = np.zeros((Y_.shape[0], env.n_actions))


    # Y[np.arange(Y_.shape[0]), Y_] = 1
    Y += 1. / (env.n_actions + 1)
    Y[np.arange(Y_.shape[0]), Y_] += 1. / (env.n_actions + 1)

    RL.mimic_learn(X, Y)

    env.after(100, run_maze)
    env.mainloop()

