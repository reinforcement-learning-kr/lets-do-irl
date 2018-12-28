import pylab
import numpy as np
import gym

n_states = 50
q_table = np.zeros((n_states, n_states, 3))
learning_rate = 0.03
gamma = 0.9
epsilon = 1.0
initial_exploration = 10000

def obs_to_state(env, state):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low) / n_states
    position = int((state[0] - env_low[0]) / env_dx[0])
    velocity = int((state[1] - env_low[1]) / env_dx[1])
    return position, velocity

def get_action(state):
    if np.random.rand() > epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[position][velocity])
    return action

def update_q_table(position, velocity, action, reward, next_position, next_velocity):
    q_1 = q_table[position][velocity][action]
    q_2 = reward + gamma * max(q_table[next_position][next_velocity])
    q_table[position][velocity][action] += learning_rate * (q_2 - q_1)

def find_policy():
    return q_table[position][velocity]


if __name__ == "__main__":
    env = gym.make('MountainCar-v0')

    scores, episodes = [], []
    steps = 0

    for episode in range(500000):
        state = env.reset()
        score = 0

        while True:
            # env.render()
            steps += 1

            position, velocity = obs_to_state(env, state)
            action = get_action(q_table[position][velocity])
            next_state, reward, done, _ = env.step(action)
            score += reward

            next_position, next_velocity = obs_to_state(env, next_state)
            update_q_table(position, velocity, action, reward, next_position, next_velocity)

            if steps > initial_exploration:
                epsilon -= 0.00005
                epsilon = max(epsilon, 0.1)

            if done:
                scores.append(score)
                episodes.append(episode)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/mountaincar_q_learning.png")
                break

        if episode % 100 == 0:
            print('{} episode | score: {:.1f}'.format(
                episode, score))
