import gym
import numpy as np

from algorithms import maxent

n_states = 625 # position - 30, velocity - 30
n_actions = 3
one_feature = 25 # number of state per one feature

feature_matrix = np.eye((n_states)) # (625, 625)
q_table = np.zeros((n_states, n_actions)) # (625, 3)

gamma = 0.9
q_learning_rate = 0.03
epochs = 10
theta_learning_rate = 0.01

def trasform_traj(env, one_feature):
    env_low = env.observation_space.low     
    env_high = env.observation_space.high   
    env_distance = (env_high - env_low) / one_feature  

    raw_trej = np.load(file="make_expert/expert_trajectories.npy")
    trajectories = np.zeros((len(raw_trej), len(raw_trej[0]), 3))

    for x in range(len(raw_trej)):
        for y in range(len(raw_trej[0])):
            position_idx = int((raw_trej[x][y][0] - env_low[0]) / env_distance[0])
            velocity_idx = int((raw_trej[x][y][1] - env_low[1]) / env_distance[1])
            state_idx = position_idx + velocity_idx * one_feature

            trajectories[x][y][0] = state_idx
            trajectories[x][y][1] = raw_trej[x][y][2] 
            trajectories[x][y][2] = raw_trej[x][y][3] 
            
    return trajectories

def idx_to_state(env, state):
    env_low = env.observation_space.low
    env_high = env.observation_space.high 
    env_distance = (env_high - env_low) / one_feature 
    position_idx = int((state[0] - env_low[0]) / env_distance[0])
    velocity_idx = int((state[1] - env_low[1]) / env_distance[1])
    state_idx = position_idx + velocity_idx * one_feature
    return state_idx

def update_q_table(state, action, reward, next_state):
    q_1 = q_table[state][action]
    q_2 = reward + gamma * max(q_table[next_state])
    q_table[state][action] += q_learning_rate * (q_2 - q_1)

def find_policy():
    return q_table


def main():
    env = gym.make('MountainCar-v0')
    trajectories = trasform_traj(env, one_feature)
    
    for episode in range(500000):
        state = env.reset()
        score = 0
        step = 0

        while True:
            # env.render()
            # print(step, "step")
            state_idx = idx_to_state(env, state)
            action = np.argmax(q_table[state_idx])
            next_state, reward, done, _ = env.step(action)
                    
            next_state_idx = idx_to_state(env, next_state)
            if step % 100 == 0 and step != 0:
                # (625,)
                irl_reward = maxent.maxent_irl(feature_matrix, n_actions, gamma, 
                                                trajectories, epochs, theta_learning_rate)
                irl_reward = irl_reward[next_state_idx]
                update_q_table(state_idx, action, irl_reward, next_state_idx)
            else:
                update_q_table(state_idx, action, reward, next_state_idx)      
            
            score += reward
            state = next_state
            step += 1

            if done:
                break

        if episode % 100 == 0:
            print('{} episode | score : {:.1f}'.format(episode, score))

if __name__ == '__main__':
    main()