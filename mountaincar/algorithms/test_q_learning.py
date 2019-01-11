import numpy as np
import gym
import random
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm

N_idx = 30

def idx_to_state(env, state):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_distance = (env_high - env_low) / N_idx
    position_idx = int((state[0] - env_low[0]) / env_distance[0])
    velocity_idx = int((state[1] - env_low[1]) / env_distance[1])
    state_idx = position_idx + velocity_idx * N_idx
    return state_idx

def plot_table(q_table, epi):

    q_table = np.reshape(q_table,(N_idx,N_idx,3))
    
    fig,ax = plt.subplots(1,3,  figsize=(30, 8))
    
    plt.title('Q_table in episode : {}'.format(epi))
    
    for a_i in range(3):
      ax[a_i].set_title("action : {}".format(a_i))
      ims = ax[a_i].imshow(q_table[:,:,a_i], cmap=cm.gray, interpolation=None, vmin=-70, vmax=0)
      ax[a_i].set_xticks(np.arange(0, N_idx-1, 3))
      ax[a_i].set_xlabel('Position')
      
      ax[a_i].set_yticks(np.arange(0, N_idx-1, 3))
    
    ax[0].set_ylabel('Velocity')
    fig.colorbar(ims, ax=ax)

    plt.draw()
    plt.savefig('Q_table in episode_{}.png'.format(epi))
    plt.close(fig)

if __name__ == '__main__':
    print(":: Start Q-learning.\n")
    gamma = 0.99
    q_learning_rate = 0.03

    n_states = N_idx**2  # position - 50, velocity - 50
    n_actions = 3
    q_table = np.zeros((n_states, n_actions))  # (2500, 3)

    # Create a new game instance.
    env = gym.make('MountainCar-v0')
    episode = 0
    epsilon = 0.9
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
            epsilon = np.maximum(0.99*epsilon, 0.05)

            if done:
                scores.append(score)
                if np.mean(scores[-min(10, len(scores)):]) > -150:
                    print(":: TOUCH DOWN EPISODE %d / SCORE %d \n" % (episode, score))
                    np.save("best_q_table", arr=q_table)
                    print(":: Complete Q-learning.\n")
                    plot_table(q_table, episode)
                    env.close()
                    sys.exit()

                episode += 1
                break

        if episode % 1000 == 0:
            print('{} episode | score: {:.1f}'.format(episode, score))
            plot_table(q_table, episode)

