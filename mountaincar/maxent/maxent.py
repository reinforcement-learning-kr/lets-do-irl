import numpy as np
from itertools import product
from train import get_q_table

def maxent_irl(feature_matrix, n_actions, gamma, demonstrations, epochs, learning_rate):
    # n_states = 400
    n_states = feature_matrix.shape[0]
    
    # Initialize weights.
    theta = -(np.random.uniform(size=(n_states,)))
    
    # Calculate the expert's feature expectations \tilde{f}. 
    expert = expert_feature_expectations(feature_matrix, demonstrations)

    # Gradient descent on theta.
    for i in range(epochs):
        print("epochs:", i)
        reward = feature_matrix.dot(theta)

        expected_svf = get_expected_svf(n_states, n_actions, reward, gamma, demonstrations)
        learner = feature_matrix.T.dot(expected_svf)
        
        gradient = expert - learner

        theta += learning_rate * gradient

    # Clip theta
    for j in range(len(theta)):
        if theta[j] > 0:
            theta[j] = 0

    return feature_matrix.dot(theta).reshape((n_states,))


def expert_feature_expectations(feature_matrix, demonstrations):
    feature_expectations = np.zeros(feature_matrix.shape[0])
    
    for demonstration in demonstrations:
        for state_idx, _, _ in demonstration:
            feature_expectations += feature_matrix[int(state_idx)]

    feature_expectations /= demonstrations.shape[0]
    return feature_expectations


def get_expected_svf(n_states, n_actions, reward, gamma, demonstrations):
    demo_num = demonstrations.shape[0]
    demo_length = demonstrations.shape[1]
    start_state_count = np.zeros(n_states)
    
    # Step 3 in Local action probability computation
    for demonstration in demonstrations:
        start_state_count[int(demonstration[0, 0])] += 1
    p_initial_state = start_state_count/demo_num
    
    # Step 4 in Forward pass
    expected_svf = np.tile(p_initial_state, (demo_length, 1)).T

    # (400, 3)
    policy = get_q_table()
    
    # Step 5 in Forward pass
    for t in range(1, demo_length):
        expected_svf[:, t] = 0
        for i, j, k in product(range(n_states), range(n_actions), range(n_states)):
            expected_svf[k, t] += expected_svf[i, t-1] * policy[i, j] * 1

    # Step 6 in Summing frequencies
    return expected_svf.sum(axis=1)