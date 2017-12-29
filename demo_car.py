# while True:
#     action = raw_input()
#     print action

import numpy as np
from pynput.keyboard import Key, Listener
import pickle
import os
import gym
import time

# create 'static' folder
if not os.path.isdir(os.path.join(os.path.dirname(__file__), 'models')):
    os.makedirs(os.path.join(os.path.dirname(__file__), 'models'))

save_path = 'models/mimic_car.pickle'

observations = []

observation = None
max_episode = 5
episode = 0

current_action = 1


def on_press(key):
    global episode
    global observations
    global observation
    global current_action
    action = str(key)
    if action == 'u\'s\'':
        action = 1
    elif action == 'u\'d\'':
        action = 2
    elif action == 'u\'a\'':
        action = 0
    else:
        action = None
    print(action)
    print('{0} pressed'.format(
        key))
    if action is not None:
      current_action = action

    # if action is not None:
    #     print('step')
    #     observation_, reward, done = env.step(action)
    #     observations.append(np.hstack((observation, action, reward, observation_)))
    #     observation = observation_
    #     env.render()
    #     if done:
    #         episode += 1
    #         if episode >= max_episode:
    #             print('game over!')
    #             print('save history..')
    #             history = np.array(observations)
    #             with open(save_path, 'wb') as f:
    #                 pickle.dump(history, f, protocol=pickle.HIGHEST_PROTOCOL)
    #             print('save successful!')
    #             with open(save_path, 'rb') as f:
    #                 verify = pickle.load(f)
    #             # to list to avoid some bug of numpy in child thread
    #             if history.tolist() == verify.tolist():
    #                 print('verify success')
    #             else:
    #                 print('failed!')
    #             env.destroy()
    #         else:
    #             observation = env.reset()


def on_release(key):
    print('{0} release'.format(
        key))
    if key == Key.esc:
        # Stop listener
        return False


def get_action():
    global observation
    observation = env.reset()
    listener = Listener(
            on_press=on_press,
            on_release=on_release)
    listener.start()


if __name__ == "__main__":
    # maze game
    env = gym.make('MountainCar-v0')
    env = env.unwrapped
    get_action()
    for ep in range(max_episode):
        observation = env.reset()
        while True:
            env.render()
            time.sleep(0.02)
            observation_, reward, done, info = env.step(current_action)
            position, velocity = observation_
            observations.append(np.hstack((observation, current_action, reward, observation_)))
            # the higher the better
            reward = abs(position - (-0.5))  # r in [0, 1]
            observation = observation_
            if done:
                break
    history = np.array(observations)
    with open(save_path, 'wb') as f:
        pickle.dump(history, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('save successful!')
    with open(save_path, 'rb') as f:
        verify = pickle.load(f)
    # to list to avoid some bug of numpy in child thread
    if history.tolist() == verify.tolist():
        print('verify success')
    else:
        print('failed!')
    print('env ended!')



