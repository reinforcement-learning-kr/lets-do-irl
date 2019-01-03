import gym
import numpy as np

from algorithms import maxent

pos_vel_state = 50
n_states = 2500 # position - 50, velocity - 50
n_actions = 3 # Left, Stop, Right

feature_matrix = np.eye((n_states)) # (2500, 2500)

# grid_State with 3 action types
q_table = np.zeros((n_states**2, n_actions)) # (2500, 3) 


gamma = 0.9
q_learning_rate = 0.03
epochs = 300
theta_learning_rate = 0.01

def conv_traj(env, pos_vel_state):
    print ("########## conv_traj(env, pos_vel_state ##########")
    #global trajectories
    env_low = env.observation_space.low     # [-1.2, -0.07]
    env_high = env.observation_space.high   # [0.6, 0.07]
    env_distance = (env_high - env_low) / pos_vel_state  # n_state = 50

    raw_trej = np.load(file="make_expert/expert_trajectories.npy")
    
    '''
    print ("### trejectories:", raw_trej)   # (state[0], state[1], action, reward) ((20, 100, 4))
    print ("### trejectories :", raw_trej[0][0][0], raw_trej[0][0][1])
    print ("### trejectories :", len(raw_trej[0][0]))
    print ("### trejectories :", len(raw_trej[0]))
    print ("### trejectories :", len(raw_trej))
    #print ("### trejectories shape:", raw_trej[1])
    '''
    
    trajectories = np.zeros((len(raw_trej), len(raw_trej[0]), 3))

    for x in range(0, len(raw_trej)):
        for y in range(0, len(raw_trej[0])):
            position_idx = int((raw_trej[x][y][0] - env_low[0]) / env_distance[0])
            velocity_idx = int((raw_trej[x][y][1] - env_low[1]) / env_distance[1])
            state_idx = position_idx + velocity_idx*pos_vel_state

            trajectories[x][y][0] = state_idx
            trajectories[x][y][1] = raw_trej[x][y][2] 
            trajectories[x][y][2] = raw_trej[x][y][3] 
            '''
            print("[x],[y]:", x ,y)
            print("state_idx:", state_idx)
            print("raw_trej[x][y][0],raw_trej[x][y][1]:", raw_trej[x][y][0] ,raw_trej[x][y][1])
            print("trajectories[x][y]:", trajectories[x][y])

    
    print ("### trejectories :", trajectories[0][0][0], trajectories[0][0][1])
    print ("### trejectories :", len(trajectories[0][0]))
    print ("### trejectories :", len(trajectories[0]))
    print ("### trejectories :", len(trajectories))
    print ("### type(trejectories) :", (trajectories.shape))
    

    print (trajectories)
    '''
    print ("########## conv_traj(env, pos_vel_state ##########")
    return trajectories

def find_policy():
    return q_table

    
def idx_to_state(env, state):
    print ("########## idx_to_state(env, state) ##########")
    env_low = env.observation_space.low     # [-1.2, -0.07]
    env_high = env.observation_space.high   # [0.6, 0.07]
    env_distance = (env_high - env_low) / pos_vel_state  # n_state = 50
    position_idx = int((state[0] - env_low[0]) / env_distance[0])
    velocity_idx = int((state[1] - env_low[1]) / env_distance[1])
    state_idx = position_idx + velocity_idx * pos_vel_state
    print ("### state_idx:", state_idx)
    print ("########## idx_to_state(env, state) ##########")
    return state_idx

def get_action(state_idx):
    print ("########## get_action(state_idx) ##########")
    print ("### state_idx:", state_idx)
    print ("### q_table[int(state_idx)]:", q_table[state_idx])
    print ("########## get_action(state_idx) ##########")
    return np.argmax(q_table[state_idx])

def update_q_table(state, action, reward, next_state):
    print ("########## update_q_table(state, action, reward, next_state) ##########")
    print ("### q_table:", q_table)
    q_1 = q_table[state][action]
    print ("### q_1:", q_1)

    q_2 = reward + gamma * max(q_table[next_state])
    q_table[state][action] += q_learning_rate * (q_2 - q_1)
    print ("########## update_q_table(state, action, reward, next_state) ##########")

def find_policy():
    return q_table

def main():
    env = gym.make('MountainCar-v0')
    trajectories = conv_traj(env, pos_vel_state)

    for episode in range(500000):
        state = env.reset()
        score = 0

        while True:
            env.render()
            print ("### state:", state)
            state = idx_to_state(env, state)
            print ("### next_state after idx_to_state:", state)
                        
            action = get_action([int(state)])
            print ("### action:", action)

            next_state, _, done, _ = env.step(action)
            print ("### next_state:", next_state)
            print ("### done:", done)
            
            next_state = idx_to_state(env, next_state)
            print ("### next_state after idx_to_state:", next_state)

            # (2500, 3, 2500)
            irl_reward = maxent.maxent_irl(feature_matrix, n_actions, q_table, gamma, 
                                trajectories, epochs, theta_learning_rate, env)  

            reward = irl_reward[state][action][next_state]
            update_q_table(state, action, reward, next_state)

            score += reward
            state = next_state

            
            if done:
                break

        if episode % 100 == 0:
            print('{} episode | score: {:.1f}'.format(
                episode, score))

if __name__ == '__main__':
    main()