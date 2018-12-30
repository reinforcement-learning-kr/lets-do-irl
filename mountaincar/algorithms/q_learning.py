import numpy as np
import gym

n_states = 50
n_actions = 3
q_table = np.zeros((n_states, n_states, n_actions))
learning_rate = 0.03
gamma = 0.9

def idx_to_state(env, state, n_states):
    env_low = env.observation_space.low 
    env_high = env.observation_space.high 
    env_distance = (env_high - env_low) / n_states 
    position = int((state[0] - env_low[0]) / env_distance[0])
    velocity = int((state[1] - env_low[1]) / env_distance[1])
    return position, velocity

def get_action(state):
    return np.argmax(q_table[position][velocity])

def update_q_table(position, velocity, action, reward, next_position, next_velocity):
    q_1 = q_table[position][velocity][action]
    q_2 = reward + gamma * max(q_table[next_position][next_velocity])
    q_table[position][velocity][action] += learning_rate * (q_2 - q_1)


if __name__ == "__main__":
    env = gym.make('MountainCar-v0')

    for episode in range(600000):
        state = env.reset()
        score = 0

        while True:
            # env.render()
            position, velocity = idx_to_state(env, state)
            action = get_action(q_table[position][velocity])
            next_state, reward, done, _ = env.step(action)

            next_position, next_velocity = idx_to_state(env, next_state)
            update_q_table(position, velocity, action, reward, next_position, next_velocity)

            score += reward
            state = next_state
            
            if done:
                break

        if episode % 100 == 0:
            print('{} episode | score: {:.1f}'.format(
                episode, score))
