"""
Deep Q network,

Using:
Tensorflow: 1.0
gym: 0.8.0
"""


import gym
import numpy as np
from RL_brain import DeepQNetwork

# from gym.envs.classic_control import rendering
# def repeat_upsample(rgb_array, k=1, l=1, err=[]):
#     # repeat kinda crashes if k/l are zero
#     if k <= 0 or l <= 0:
#         if not err:
#             print "Number of repeats must be larger than 0, k: {}, l: {}, returning default array!".format(k, l)
#             err.append('logged')
#         return rgb_array
#
#     # repeat the pixels k times along the y axis and l times along the x axis
#     # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)
#
#     return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)
#
# viewer = rendering.SimpleImageViewer()

env = gym.make('MsPacman-ram-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)
print(env.observation_space.shape)

RL = DeepQNetwork(n_actions=env.action_space.n, n_features=env.observation_space.shape[0], learning_rate=0.001, e_greedy=0.9,
                  replace_target_iter=300, memory_size=3000,
                  e_greedy_increment=0.0002,)

total_steps = 0


for i_episode in range(1000):

    observation = env.reset()
    ep_r = 0
    while True:
        # rgb = env.render('rgb_array')
        # upscaled = repeat_upsample(rgb, 3, 3)
        # viewer.imshow(upscaled)
        print('epoch {}'.format(i_episode))
        if i_episode > 500:
            env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        # position, velocity = observation_
        #
        # # the higher the better
        # reward = abs(position - (-0.5))     # r in [0, 1]

        reward = reward

        RL.store_transition(observation, action, reward, observation_)

        if total_steps > 1000:
            RL.learn()

        ep_r += reward
        if done:
            # get = '| Get' if observation_[0] >= env.unwrapped.goal_position else '| ----'
            # print('Epi: ', i_episode,
            #       get,
            #       '| Ep_r: ', round(ep_r, 4),
            #       '| Epsilon: ', round(RL.epsilon, 2))
            break

        observation = observation_
        total_steps += 1

RL.plot_cost()
