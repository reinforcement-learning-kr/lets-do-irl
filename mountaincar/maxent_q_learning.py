import gym
import numpy as np

import make_expert
from algorithms import maxent

n_states = 2500 
n_actions = 3

feature_matrix = np.eye((n_states)) # (2500, 2500)
q_table = np.zeros((n_states, n_actions)) # (2500, 3)

gamma = 0.9
q_learning_rate = 0.03
epochs = 300
theta_learning_rate = 0.01

# (20, 100, 4)
trejectories = np.load(file="make_expert/expert_trajectories.npy")

def idx_to_state(env, state):
    env_low = env.observation_space.low # [-1.2  -0.07]
    env_high = env.observation_space.high # [0.6  0.07]
    env_distance = (env_high - env_low) / 50 
    position_idx = int((state[0] - env_low[0]) / env_distance[0])
    velocity_idx = int((state[1] - env_low[1]) / env_distance[1])
    state_idx = position_idx + velocity_idx*50
    return state_idx

def get_action(state_idx):
    return np.argmax(q_table[state_idx])

def update_q_table(state, action, reward, next_state):
    q_1 = q_table[state][action]
    q_2 = reward + gamma * max(q_table[next_state])
    q_table[state][action] += q_learning_rate * (q_2 - q_1)

def find_policy():
    return q_table


def main():
    env = gym.make('MountainCar-v0')

    for episode in range(500000):
        state = env.reset()
        score = 0

        while True:
            # env.render()
            state_idx = idx_to_state(env, state)
            action = get_action(q_table[state_idx])
            next_state, _, done, _ = env.step(action)
            
            # (2500,)
            irl_reward = maxent.maxent_irl(grid_states, n_actions, gamma, 
                                trajectories, epochs, theta_learning_rate, env)         
            next_state_idx = idx_to_state(env, next_state)                 
            reward = irl_reward[next_state_idx]
            update_q_table(state, action, reward, next_state)

            score += reward
            state = next_state
            
            if done:
                break

        if episode % 100 == 0:
            print('{} episode | score: {:.1f}'.format(
                episode, score))

if __name__ == '__main__':
    main()