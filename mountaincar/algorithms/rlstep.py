import numpy as np
import random
import gym
from algorithms.feature_estimate import FeatureEstimate

def idx_to_state(env, state):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_distance = (env_high - env_low) / 50
    position_idx = int((state[0] - env_low[0]) / env_distance[0])
    velocity_idx = int((state[1] - env_low[1]) / env_distance[1])
    state_idx = position_idx + velocity_idx * 50
    return state_idx

def q_learning(weights, i):
    print("Iternation %d :: Start Q-learning.\n" % (i))
    gamma = 0.99
    q_learning_rate = 0.03

    n_states = 2500  # position - 50, velocity - 50
    n_actions = 3
    q_table = np.zeros((n_states, n_actions))  # (2500, 3)

    # Create a new game instance.
    env = gym.make('MountainCar-v0')

    # Feature estimator
    feature_estimate = FeatureEstimate(env, len(weights))

    # Run the frames.
    count = 0
    while count < 10:
        state = env.reset()
        score = 0
        epsilon = 0.1
        episode = 1


        while True:
            #env.render()
            state_idx = idx_to_state(env, state)
            if random.random() < epsilon:
                action = np.random.randint(0, 2)  # random #3
            else:
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


            if state[0] >= 0.5:
                count += 1
                print("TOUCH DOWN :: %d \n" % (count))

            if done:
                episode += 1
                break

        if episode % 100 == 0:
            print('{} episode | score: {:.1f}'.format(episode, score))

    print("Iternation %d :: Complete Q-learning.\n" % (i))
    return q_table
