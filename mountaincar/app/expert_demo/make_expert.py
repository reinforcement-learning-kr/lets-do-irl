import gym
import readchar
import numpy as np

# MACROS
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

for episode in range(20): # n_trajectories : 20
    trajectory = []
    step = 0

    env.reset()
    print("episode_step", episode_step)

    while True: 
        env.render()
        print("step", step)

        key = readchar.readkey()
        if key not in arrow_keys.keys():
            break

        action = arrow_keys[key]
        state, reward, done, _ = env.step(action)

        if state[0] >= env.env.goal_position and step > 129: # trajectory_length : 130
            break

        trajectory.append((state[0], state[1], action))
        step += 1

    trajectory_numpy = np.array(trajectory, float)
    print("trajectory_numpy.shape", trajectory_numpy.shape)
    episode_step += 1
    trajectories.append(trajectory)

np_trajectories = np.array(trajectories, float)
print("np_trajectories.shape", np_trajectories.shape)

np.save("expert_demo", arr=np_trajectories)