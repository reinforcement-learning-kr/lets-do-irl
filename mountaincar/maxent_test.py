import gym
import pylab
import numpy as np

q_table = np.load(file="results/maxent_q_table/maxent_200_epoch_9000_epi.npy") # (400, 3)
one_feature = 20 # number of state per one feature
gamma = 0.9
q_learning_rate = 0.03

def idx_to_state(env, state):
    env_low = env.observation_space.low
    env_high = env.observation_space.high 
    env_distance = (env_high - env_low) / one_feature 
    position_idx = int((state[0] - env_low[0]) / env_distance[0])
    velocity_idx = int((state[1] - env_low[1]) / env_distance[1])
    state_idx = position_idx + velocity_idx * one_feature
    return state_idx
    
def main():
    env = gym.make('MountainCar-v0')
    
    episodes, scores = [], []

    for episode in range(1000):
        state = env.reset()
        score = 0

        while True:
            env.render()
            state_idx = idx_to_state(env, state)
            action = np.argmax(q_table[state_idx])
            next_state, reward, done, _ = env.step(action)
            
            score += reward
            state = next_state
            
            if done:
                scores.append(score)
                episodes.append(episode)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./learning_curves/maxent_test.png")
                break

        if episode % 1 == 0:
            #score_avg = np.mean(scores)
            print('{} episode score is {:.2f}'.format(episode, score))

if __name__ == '__main__':
    main()
    