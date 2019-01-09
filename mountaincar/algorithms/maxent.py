import numpy as np
from itertools import product
from maxent_train import find_policy

def maxent_irl(feature_matrix, n_actions, gamma, trajectories, epochs, learning_rate):
    # 400
    n_states = feature_matrix.shape[0]
    
    # Initialise weights.
    theta = -(np.random.uniform(size=(n_states,)))
    
    # Calculate the feature expectations \tilde{f}. 
    feature_expectations = find_feature_expectations(feature_matrix, trajectories)

    # Gradient descent on theta.
    for i in range(epochs):
        reward = feature_matrix.dot(theta)

        expected_svf = find_expected_svf(n_states, n_actions, reward, gamma, trajectories)
        gradient = feature_expectations - feature_matrix.T.dot(expected_svf)
        theta += learning_rate * gradient

    for j in range(len(theta)):
        if theta[j] > 0:
            theta[j] = 0

    return feature_matrix.dot(theta).reshape((n_states,))

def find_feature_expectations(feature_matrix, trajectories):
    """ Calculate the feature expectations \tilde{f}. """
    feature_expectations = np.zeros(feature_matrix.shape[0])
    
    for trajectory in trajectories:
        for state_idx, _, _ in trajectory:
            feature_expectations += feature_matrix[int(state_idx)]

    feature_expectations /= trajectories.shape[0]
    return feature_expectations

def find_expected_svf(n_states, n_actions, reward, gamma, trajectories):
    """ Algorithm 1, Expected Edge Frequency Calculation """
    n_trajectories = trajectories.shape[0] # 20
    trajectory_length = trajectories.shape[1] # 100
    
    # Step 3 in Local action probability computation 
    # & Step 4 in Forward pass 
    start_state_count = np.zeros(n_states)
    for trajectory in trajectories:
        start_state_count[int(trajectory[0, 0])] += 1
    
    p_initial_state = start_state_count/n_trajectories
    
    expected_svf = np.tile(p_initial_state, (trajectory_length, 1)).T

    # (400, 3)
    policy = find_policy()
    
    # Step 5 in Forward pass
    for t in range(1, trajectory_length):
        expected_svf[:, t] = 0
        for i, j, k in product(range(n_states), range(n_actions), range(n_states)):
            expected_svf[k, t] += expected_svf[i, t-1] * policy[i, j]

    # Step 6 in Summi􏰁ng frequencies
    return expected_svf.sum(axis=1)