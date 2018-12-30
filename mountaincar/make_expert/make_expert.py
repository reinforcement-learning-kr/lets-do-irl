import gym
import readchar
import numpy as np

# # MACROS
Push_Left = 0
No_Push = 1
Push_Right = 2

# Key mapping
arrow_keys = {
    '\x1b[D': Push_Left,
    '\x1b[B': No_Push,
    '\x1b[C': Push_Right}

env = gym.make('MountainCar-v0')

trajectories = []
episode_step = 0

for episode in range(20): # n_trajectories
    trajectory = []
    step = 0

    env.reset()
    print("episode_step", episode_step)

    for _ in range(100): # trajectory_length
        env.render()
        print("step", step)

        key = readchar.readkey()
        if key not in arrow_keys.keys():
            break

        action = arrow_keys[key]
        state, reward, done, _ = env.step(action)
        # print("State : {} | Action : {} | Reward : {} | Step : {}".format(
        #         state, action, reward, step))
        step += 1

        trajectory.append((state[0], state[1], action, reward))

        if done:
            break

    episode_step += 1
    trajectories.append(trajectory)
    
np_trajectories = np.array(trajectories, float)
print("np_trajectories.shape", np_trajectories.shape)

np.save("mountaincar_expert", arr=np_trajectories)