import numpy as np
import gym
import random
import sys

def idx_to_state(env, state):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_distance = (env_high - env_low) / 50
    position_idx = int((state[0] - env_low[0]) / env_distance[0])
    velocity_idx = int((state[1] - env_low[1]) / env_distance[1])
    state_idx = position_idx + velocity_idx * 50
    return state_idx

if __name__ == '__main__':
    print(":: Start Q-learning.\n")
    gamma = 0.99
    q_learning_rate = 0.03

    n_states = 2500  # position - 50, velocity - 50
    n_actions = 3
    q_table = np.zeros((n_states, n_actions))  # (2500, 3)

    # Create a new game instance.
    env = gym.make('MountainCar-v0')
    episode = 0
    epsilon = 0.1
    scores = []
    #while count < 10:
    while True:
        state = env.reset()
        score = 0


        while True:
            #env.render()
            state_idx = idx_to_state(env, state)

            if random.random() < epsilon:
                action = np.random.randint(0, 2)  # random #3
            else:
                action = np.argmax(q_table[state_idx])

            next_state, reward, done, _ = env.step(action)
            next_state_idx = idx_to_state(env, next_state)

            # Update Q-table
            q_1 = q_table[state_idx][action]
            q_2 = reward + gamma * max(q_table[next_state_idx])
            q_table[state_idx][action] += q_learning_rate * (q_2 - q_1)

            score += reward
            state = next_state



            if done:
                scores.append(score)
                if np.mean(scores[-min(10, len(scores)):]) > -110:
                    print(":: TOUCH DOWN EPISODE %d / SCORE %d \n" % (episode, score))
                    np.save("best_q_table", arr=q_table)
                    print(":: Complete Q-learning.\n")
                    env.close()
                    sys.exit()

                episode += 1
                break

        if episode % 1000 == 0:
            print('{} episode | score: {:.1f}'.format(episode, score))

