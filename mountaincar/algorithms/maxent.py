import numpy as np
from itertools import product
import q_learning

def maxent_irl(grid_states, n_actions, q_table, gamma, 
                trajectories, epochs, learning_rate, env):
    reward = np.zeros((grid_states, n_actions, grid_states))
    n_states = feature_matrix.shape[0]
    
    # Calculate the feature expectations \tilde{f}. 
    feature_expectations = find_feature_expectations(feature_matrix, trajectories)

    # Gradient descent on theta.
    for i in range(epochs):
        reward = feature_matrix.dot(theta)

        expected_svf = find_expected_svf(n_states, n_actions, reward, gamma, q_table, trajectories)
        gradient = feature_expectations - feature_matrix.T.dot(expected_svf)
        reward += learning_rate * gradient
    return reward # (2500, 3, 2500)


def find_feature_expectations(feature_matrix, trajectories):
    """ Calculate the feature expectations \tilde{f}. """
    feature_expectations = np.zeros(feature_matrix.shape[0])
    
    # trajectories = (20, 100, 4)
    for trajectory in trajectories:
        for state, _, _ in trajectory: # state, action, reward
            idx_to_state
            feature_expectations += feature_matrix[state]

    feature_expectations /= trajectories.shape[0]
    return feature_expectations


def find_expected_svf(n_states, n_actions, reward, gamma, policy, trajectories):
    """ Algorithm 1, Expected Edge Frequency Calculation """
    n_trajectories = trajectories.shape[0] # 20
    trajectory_length = trajectories.shape[1] # 100
    
    # Step 3 in Local action probability computation & Step 4 in Forward pass 
    start_state_count = np.zeros((n_states, n_states))
    for trajectory in trajectories:
        state = [trajectory[0][0], trajectory[0][1]]
        position_idx, velocity_idx = q_learning.idx_to_state(env, state, n_states)
        start_state_count[position_idx, velocity_idx] += 1
    p_initial_state = start_state_count/n_trajectories
    
    expected_svf = np.tile(p_initial_state, (trajectory_length, 1)).T

    # Step 5 in Forward pass
    for t in range(1, trajectory_length):
        expected_svf[:, t] = 0
        for i, j, k in product(range(n_states), range(n_actions), range(n_states)):
            expected_svf[k, t] += (expected_svf[i, t-1] *
                                  policy[i, j] # Stochastic policy
    
    # Step 6 in Summi􏰁ng frequencies
    return expected_svf.sum(axis=1)