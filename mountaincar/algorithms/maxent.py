from itertools import product

import numpy as np
import numpy.random as rn

from . import value_iteration

def maxent_irl(feature_matrix, n_actions, discount, transition_probability,
        trajectories, epochs, learning_rate):
    """
    Find the reward function for the given trajectories.

    feature_matrix: Matrix with the nth row representing the nth state. NumPy
        array with shape (N, D) where N is the number of states and D is the
        dimensionality of the state.
    n_actions: Number of actions A. int.
    discount: Discount factor of the MDP. float.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    epochs: Number of gradient descent steps. int.
    learning_rate: Gradient descent learning rate. float.
    -> Reward vector with shape (N,).
    """

    n_states, d_states = feature_matrix.shape

    # Initialise weights.
    alpha = rn.uniform(size=(d_states,))

    # Calculate the feature expectations \tilde{phi}.
    feature_expectations = find_feature_expectations(feature_matrix,
                                                     trajectories)

    # Gradient descent on alpha.
    for i in range(epochs):
        # print("i: {}".format(i))
        r = feature_matrix.dot(alpha)
        expected_svf = find_expected_svf(n_states, r, n_actions, discount,
                                         transition_probability, trajectories)
        grad = feature_expectations - feature_matrix.T.dot(expected_svf)

        alpha += learning_rate * grad

    return feature_matrix.dot(alpha).reshape((n_states,))

def find_feature_expectations(feature_matrix, trajectories):
    """
    Find the feature expectations for the given trajectories. This is the
    average path feature vector.

    feature_matrix: Matrix with the nth row representing the nth state. NumPy
        array with shape (N, D) where N is the number of states and D is the
        dimensionality of the state.
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> Feature expectations vector with shape (D,).
    """

    feature_expectations = np.zeros(feature_matrix.shape[1])

    for trajectory in trajectories:
        for state, _, _ in trajectory:
            feature_expectations += feature_matrix[state]

    feature_expectations /= trajectories.shape[0]

    return feature_expectations

def find_expected_svf(n_states, r, n_actions, discount,
                      transition_probability, trajectories):
    """
    Find the expected state visitation frequencies using algorithm 1 from
    Ziebart et al. 2008.

    n_states: Number of states N. int.
    alpha: Reward. NumPy array with shape (N,).
    n_actions: Number of actions A. int.
    discount: Discount factor of the MDP. float.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> Expected state visitation frequencies vector with shape (N,).
    """

    n_trajectories = trajectories.shape[0]
    trajectory_length = trajectories.shape[1]

    # policy = find_policy(n_states, r, n_actions, discount,
    #                                 transition_probability)
    policy = value_iteration.find_policy(n_states, n_actions,
                                         transition_probability, r, discount)
    # print("policy", policy)

    # [[0.30502566 0.33856828 0.17820303 0.17820303]
    # [0.34415869 0.32322055 0.12266247 0.20995829]
    # [0.25322372 0.41723365 0.12486571 0.20467693]
    # [0.26186551 0.33773452 0.17897471 0.22142526]
    # [0.2359474  0.32859549 0.19950971 0.2359474 ]
    # [0.32111827 0.32548617 0.2315309  0.12186466]
    # [0.45512654 0.2574835  0.15118402 0.13620593]
    # [0.3538371  0.28255481 0.17610018 0.18750791]
    # [0.28841653 0.24793443 0.28853435 0.17511469]
    # [0.24945037 0.34042006 0.23101218 0.17911739]
    # [0.29574031 0.28649997 0.24411282 0.17364689]
    # [0.31174274 0.29703179 0.19693412 0.19429134]
    # [0.27650811 0.21965688 0.18204796 0.32178705]
    # [0.36020443 0.20016277 0.19519475 0.24443804]
    # [0.25579606 0.39563252 0.16113117 0.18744025]
    # [0.31398416 0.23352178 0.24432041 0.20817365]
    # [0.28854396 0.24064708 0.23166854 0.23914043]
    # [0.23090938 0.32936016 0.21455222 0.22517824]
    # [0.43777772 0.24228897 0.14163747 0.17829584]
    # [0.2333672  0.53190456 0.08384476 0.15088348]
    # [0.26295232 0.24195305 0.24195305 0.25314157]
    # [0.3755949  0.19776397 0.18197062 0.24467051]
    # [0.33354793 0.30884734 0.16261903 0.1949857 ]
    # [0.61527437 0.14940149 0.13833769 0.09698645]
    # [0.37291729 0.37291729 0.09055212 0.1636133 ]]

    start_state_count = np.zeros(n_states)
    for trajectory in trajectories:
        start_state_count[trajectory[0, 0]] += 1
    p_start_state = start_state_count/n_trajectories

    expected_svf = np.tile(p_start_state, (trajectory_length, 1)).T
    for t in range(1, trajectory_length):
        expected_svf[:, t] = 0
        for i, j, k in product(range(n_states), range(n_actions), range(n_states)):
            expected_svf[k, t] += (expected_svf[i, t-1] *
                                  policy[i, j] * # Stochastic policy
                                  transition_probability[i, j, k])

    return expected_svf.sum(axis=1)
