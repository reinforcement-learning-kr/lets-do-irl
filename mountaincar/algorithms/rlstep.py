import numpy as np
import random
import csv
import os.path
import timeit
import gym
from algorithms.feature_estimate import FeatureEstimate

NUM_INPUT = 50
GAMMA = 0.9  # Forgetting.
TUNING = False  # If False, just use arbitrary, pre-selected params.
TRAIN_FRAMES = 100000 # to train for 100K frames in total

def idx_to_state(env, state):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_distance = (env_high - env_low) / 50
    position_idx = int((state[0] - env_low[0]) / env_distance[0])
    velocity_idx = int((state[1] - env_low[1]) / env_distance[1])
    state_idx = position_idx + velocity_idx * 50
    return state_idx

def q_learning(weights, i):

    gamma = 0.9
    q_learning_rate = 0.03

    n_states = 2500  # position - 50, velocity - 50
    n_actions = 3
    q_table = np.zeros((n_states, n_actions))  # (2500, 3)

    # Create a new game instance.
    env = gym.make('MountainCar-v0')

    # Get initial state by doing nothing and getting the state.
    state = env.reset()

    # Feature estimator
    feature_estimate = FeatureEstimate(env, len(weights))

    # Let's time it.
    start_time = timeit.default_timer()

    # Run the frames.
    for episode in range(1000):
        state = env.reset()
        score = 0

        while True:
            # env.render()
            state_idx = idx_to_state(env, state)
            action = np.argmax(q_table[state_idx])
            next_state, _, done, _ = env.step(action)

            next_state_idx = idx_to_state(env, next_state)
            features = feature_estimate.get_features(next_state)
            irl_reward = np.dot(weights, features)

            # Update Q-table
            q_1 = q_table[state_idx][action]
            q_2 = irl_reward + gamma * max(q_table[next_state_idx])
            q_table[state_idx][action] += q_learning_rate * (q_2 - q_1)

            score += irl_reward
            state = next_state

            if done:
                break

        if episode % 100 == 0:
            print('{} episode | score: {:.1f}'.format(episode, score))

    print("Complete Q-learning - %d" % (i))
    return q_table
