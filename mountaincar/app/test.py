import numpy as np
import gym
import random
import sys
import cvxpy as cp

N_idx = 20
F_idx = 4
GAMMA = 0.99

def idx_to_state(env, state):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_distance = (env_high - env_low) / N_idx
    position_idx = int((state[0] - env_low[0]) / env_distance[0])
    velocity_idx = int((state[1] - env_low[1]) / env_distance[1])
    state_idx = position_idx + velocity_idx * N_idx
    return state_idx


if __name__ == '__main__':
    print(":: Testing APP-learning.\n")
    
    # Load the agent
    n_states = N_idx**2  # position - 20, velocity - 20
    n_actions = 3
    q_table = np.load(file="results/app_q_table.npy")

    # Create a new game instance.
    env = gym.make('MountainCar-v0')
    n_episode = 10 # test the agent 10times
    scores = []

    for ep in range(n_episode):
        state = env.reset()
        score = 0

        while True:
            # Render the play
            env.render()

            state_idx = idx_to_state(env, state)

            action = np.argmax(q_table[state_idx])

            next_state, reward, done, _ = env.step(action)
            next_state_idx = idx_to_state(env, next_state)

            score += reward
            state = next_state

            if done:
                print('{} episode | score: {:.1f}'.format(ep + 1, score))
                
                break

    env.close()
    sys.exit()