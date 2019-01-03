"""
Once a Q-table is learned, use this to play it. that is run/exploit a policy to get the feature expectations of the policy
"""

import numpy as np
import sys
import gym
from algorithms.feature_estimate import FeatureEstimate

TRAJECTORIES = 100

NUM_STATES = 8
GAMMA = 0.9

def idx_to_state(env, state):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_distance = (env_high - env_low) / 50
    position_idx = int((state[0] - env_low[0]) / env_distance[0])
    velocity_idx = int((state[1] - env_low[1]) / env_distance[1])
    state_idx = position_idx + velocity_idx * 50
    return state_idx

def calcNewFE(env, q_table, weights):
    state = env.reset()
    featureExpectations = np.zeros(len(weights))
    feature_estimate = FeatureEstimate(env, len(weights))
    car_steps = 0
    while True:
        car_steps += 1

        # Choose action.
        state_idx = idx_to_state(env, state)
        action = (np.argmax(q_table[state_idx]))

        # Take action.
        next_state, _, done, _ = env.step(action)

        features = feature_estimate.get_features(next_state)
        featureExpectations += (GAMMA**(car_steps))*np.array(features)

        if car_steps % TRAJECTORIES == 0 or done == True:
            print("Current steps: %d trajectories." % car_steps)
            break

    return featureExpectations

def randomFE(num_features):
    return np.random.normal(size=num_features)

def expertFE(env, trajectories, num_features):
    featureExpectations = np.zeros(num_features)
    feature_estimate = FeatureEstimate(env, num_features)
    for m in range(len(trajectories)):
        for car_steps in range(len(trajectories)):
            state = trajectories[m][car_steps]
            features = feature_estimate.get_features(state)
            featureExpectations += (GAMMA**(car_steps))*np.array(features)
    featureExpectations = featureExpectations / len(trajectories)
    return featureExpectations