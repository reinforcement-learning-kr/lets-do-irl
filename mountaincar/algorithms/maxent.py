import numpy as np
from itertools import product
import numpy.random as rn

import maxent_q_learning

def maxent_irl(feature_matrix, n_actions, q_table, gamma, 
                trajectories, epochs, learning_rate, env):
    print ("########## maxent_irl(feature_matrix, n_actions, q_table, gamma,trajectories, epochs, learning_rate, env) ##########")
    
    print ("### feature_matrix:", feature_matrix)
    print ("### n_actions:", n_actions)
    
    n_states, d_states = feature_matrix.shape
    print ("### n_states:", n_states)
    print ("### d_states:", d_states)


    # Initialise weights.
    theta = np.random.uniform(size=(n_states,))
    print ("### theta:", theta)
        
    print ("### trajectories:", trajectories)
    # Calculate the feature expectations \tilde{f}. 
    feature_expectations = find_feature_expectations(feature_matrix, trajectories, env)
    print ("### feature_expectations:", feature_expectations)

    # Gradient descent on theta.
    for i in range(epochs):
        theta = feature_matrix.dot(theta)

        expected_svf = find_expected_svf(n_states, n_actions, theta, gamma, trajectories)
        gradient = feature_expectations - feature_matrix.T.dot(expected_svf)
        theta += learning_rate * gradient
    return theta # (2500, 3, 2500)
    print ("########## maxent_irl(grid_states, n_actions, q_table, gamma,trajectories, epochs, learning_rate, env) ##########")
    


def find_feature_expectations(feature_matrix, trajectories, env):
    print ("##########find_feature_expectations(feature_matrix, trajectories, env) ##########")
    
    """ Calculate the feature expectations \tilde{f}. """
    feature_expectations = np.zeros(feature_matrix.shape[1])
    print ("feature_expectations", feature_expectations)
    
    # trajectories = (20, 100, 4)
    for trajectory in trajectories:
        #print ("trajectory", trajectory)
        for state_idx, _, _ in trajectory: # state, action, reward
            feature_expectations += feature_matrix[int(state_idx)]

    feature_expectations /= trajectories.shape[0]
    print ("##########find_feature_expectations(feature_matrix, trajectories, env) ##########")
    return feature_expectations


def find_expected_svf(n_states, n_actions, reward, gamma, trajectories):
    print ("##########find_expected_svf(n_states, n_actions, reward, gamma, trajectories) ##########")
    """ Algorithm 1, Expected Edge Frequency Calculation """
    n_trajectories = trajectories.shape[0] # 20
    trajectory_length = trajectories.shape[1] # 100

    
    
    # Step 3 in Local action probability computation 
    # & Step 4 in Forward pass 
    start_state_count = np.zeros(n_states)

    for trajectory in trajectories:
        print("trajectory[0, 0]:", trajectory[0, 0])
        start_state_count[int(trajectory[0, 0])] += 1
    p_initial_state = start_state_count/n_trajectories
    
    expected_svf = np.tile(p_initial_state, (trajectory_length, 1)).T

    # (2500, 3)
    policy = maxent_q_learning.find_policy()

    # Step 5 in Forward pass
    for t in range(1, trajectory_length):
        expected_svf[:, t] = 0
        for i, j, k in product(range(n_states), range(n_actions), range(n_states)):
            print("[t][i][j][k]:", t, i, j, k)
            print("expected_svf[k, t]:", expected_svf[k, t])
            expected_svf[k, t] += expected_svf[i, t-1] * policy[i, j] # Stochastic policy
    
    # Step 6 in Summi􏰁ng frequencies
    print ("##########find_expected_svf(n_states, n_actions, reward, gamma, trajectories) ##########")
    
    return expected_svf.sum(axis=1)
