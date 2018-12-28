import pylab
import numpy as np
import gym

class QLearning:
    def __init__(self):
        self.learning_rate = 0.01
        self.gamma = 0.9
        self.epsilon = 1.0
        self.initial_exploration = 10000
        self.n_states = 50
        self.q_table = np.zeros((self.n_states, self.n_states, 3))

    def obs_to_state(self, env, state):
        env_low = env.observation_space.low
        env_high = env.observation_space.high
        env_dx = (env_high - env_low) / self.n_states
        position = int((state[0] - env_low[0]) / env_dx[0])
        velocity = int((state[1] - env_low[1]) / env_dx[1])
        return position, velocity

    def get_action(self, state):
        if np.random.rand() > self.epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(self.q_table[position][velocity])
        return action

    def update_q_table(self, position, velocity, action, reward, next_position, next_velocity):
        q_1 = self.q_table[position][velocity][action]
        q_2 = reward + self.gamma * max(self.q_table[next_position][next_velocity])
        self.q_table[position][velocity][action] += self.learning_rate * (q_2 - q_1)
    

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    agent = QLearning()

    scores, episodes = [], []
    steps = 0

    for episode in range(500000):
        state = env.reset()
        score = 0

        while True:
            # env.render()
            steps += 1

            position, velocity = agent.obs_to_state(env, state)
            action = agent.get_action(agent.q_table[position][velocity])
            next_state, reward, done, _ = env.step(action)
            score += reward

            next_position, next_velocity = agent.obs_to_state(env, next_state)
            agent.update_q_table(position, velocity, action, reward, next_position, next_velocity)

            if steps > agent.initial_exploration:
                agent.epsilon -= 0.00005
                agent.epsilon = max(agent.epsilon, 0.1)

            if done:
                scores.append(score)
                episodes.append(episode)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/mountaincar_q_learning.png")
                break

        if episode % 100 == 0:
            print('{} episode | score: {:.1f}'.format(
                episode, score))

    print(agent.q_table)
