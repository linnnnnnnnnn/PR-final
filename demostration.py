# while True:
#     action = raw_input()
#     print action

from maze_env import Maze
import numpy as np
from pynput.keyboard import Key, Listener
import pickle

save_path = 'mimic.pickle'

observations = []

observation = None
max_episode = 2
episode = 0


def on_press(key):
    global episode
    global observations
    global observation
    action = str(key)
    if action == 'u\'w\'':
        action = 0
    elif action == 'u\'s\'':
        action = 1
    elif action == 'u\'d\'':
        action = 2
    elif action == 'u\'a\'':
        action = 3
    else:
        action = None
    print(action)
    print('{0} pressed'.format(
        key))

    if action is not None:
        print('step')
        observation_, reward, done = env.step(action)
        observations.append(np.hstack((observation, action, reward, observation_)))
        observation = observation_
        env.render()
        if done:
            episode += 1
            if episode >= max_episode:
                print('game over!')
                print('save history..')
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
                env.destroy()
            else:
                observation = env.reset()



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
    env = Maze()

    env.after(100, get_action)
    env.mainloop()
    print('env started!')

    # Collect events until released



