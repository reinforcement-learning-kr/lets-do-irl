import numpy as np
import gym

import make_expert
from algorithms import q_learning, maxent

def main(env):
    n_states = 50 
    n_actions = 3
    q_learning_rate
    gamma = 0.9
    epochs = 300
    theta_learning_rate = 0.01
    
    trejectories = np.load(file="make_expert/expert_trajectories.npy")
    # trajectories = (20, 100, 4)
    print("trejectories.shape", trejectories.shape)
    
    # feature_matrix = (50, 50)
    feature_matrix = np.eye((n_states), dtype=int)
    print("feature_matrix", feature_matrix)
    print("feature_matrix.shape", feature_matrix.shape)

    q_table = np.zeros((n_states, n_states, n_actions))
    # print("q_table", q_table)
    print("q_table.shape", q_table.shape)

    reward = maxent.maxent_irl(feature_matrix, n_actions, gamma, 
                                trajectories, epochs, theta_learning_rate, env)

    return reward

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    main(env)

    for episode in range(600000):
        state = env.reset()
        score = 0

        while True:
            # env.render()
            position, velocity = idx_to_state(env, state)
            action = get_action(q_table[position][velocity])
            next_state, reward, done, _ = env.step(action)

            next_position, next_velocity = idx_to_state(env, next_state)
            update_q_table(position, velocity, action, reward, next_position, next_velocity)

            score += reward
            state = next_state
            
            if done:
                break

        if episode % 100 == 0:
            print('{} episode | score: {:.1f}'.format(
                episode, score))
    

