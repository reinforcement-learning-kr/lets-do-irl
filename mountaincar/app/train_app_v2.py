import numpy as np
import gym
import random
import sys
import cvxpy as cp

N_idx = 30
F_idx = 4
GAMMA = 0.99

trejectories = np.load(file="make_expert/expert_trajectories.npy")


def idx_to_state(env, state):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_distance = (env_high - env_low) / N_idx
    position_idx = int((state[0] - env_low[0]) / env_distance[0])
    velocity_idx = int((state[1] - env_low[1]) / env_distance[1])
    state_idx = position_idx + velocity_idx * N_idx
    return state_idx

def randomFE(num_features):
    return np.random.normal(size=num_features)

def expertFE(env, trajectories, num_features):
    featureExpectations = np.zeros(num_features)
    feature_estimate = FeatureEstimate(env, num_features)
    for m in range(len(trajectories)):
        for car_steps in range(len(trajectories[0])):
            state = trajectories[m][car_steps]
            features = feature_estimate.get_features(state)
            featureExpectations += (GAMMA**(car_steps))*np.array(features)
    featureExpectations = featureExpectations / len(trajectories)
    return featureExpectations

class FeatureEstimate:
    def __init__(self, env, num_features):
        self.env = env
        self.num_features = num_features
        self.feature = np.ones(self.num_features)

    def gaussian(self, x, mu):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(1., 2.)))

    def get_features(self, state):
        env_low = self.env.observation_space.low
        env_high = self.env.observation_space.high
        env_distance = (env_high - env_low) / (self.num_features - 1)

        for i in range(int(self.num_features/2)):
          self.feature[i] = self.gaussian(state[0], env_low[0] + i * env_distance[0])
          self.feature[i+int(self.num_features/2)] = self.gaussian(state[1], env_low[1] + i * env_distance[1])

        return self.feature

def QP_optimizer(expertList, agentList):
    w=cp.Variable(4)
    b=cp.Variable(1)
        
    policyMat = agentList
    ExpertMat = expertList
        
    #constraints = [ExpertMat*w + b >= 1, policyMat*w + b <= -1]
    constraints = [(ExpertMat-policyMat)*w >= 2]
        
    obj = cp.Minimize(cp.norm(w))
    prob = cp.Problem(obj, constraints)

    prob.solve()
    print("status:", prob.status)
    print("optimal value", prob.value)
    #print("optimal var", w.value)

    weights = np.squeeze(np.asarray(w.value))
    #bias = np.squeeze(np.asarray(b.value))
    bias = 0
    return weights, bias

def add_policyList(policyList, FE):
    #agent의 RL과정이 끝난 후 나온 FE를 list에 저장
    policyList = np.vstack([policyList, FE])
    return policyList

def calc_FE(env, qtable):
    featureExpectations = np.zeros(4)
    feature_estimate = FeatureEstimate(env, 4)
    scoreList = []
    
    for m in range(20):
        state = env.reset()
        car_steps = 0
        score = 0
        done = False
        
        while not done and (car_steps<=120):
            car_steps += 1

            # Choose action.
            state_idx = idx_to_state(env, state)
            action = (np.argmax(q_table[state_idx]))

            # Take action.
            next_state, r, done, _ = env.step(action)
            # calculate FE
            features = feature_estimate.get_features(next_state)
            featureExpectations += (GAMMA**(car_steps))*np.array(features)

            score+=r
            state = next_state

        scoreList.append(score)
    
    featureExpectations = featureExpectations/20
    print("avg_score:", np.mean(scoreList))
    print("\n")
    print("current_FE:", featureExpectations)

    return featureExpectations

def play(env, qtable):
    scoreList = []
    
    for m in range(10):
        state = env.reset()
        score = 0
        done = False
        
        while not done:
            env.render()
            
            # Choose action.
            state_idx = idx_to_state(env, state)
            action = (np.argmax(q_table[state_idx]))

            # Take action.
            next_state, r, done, _ = env.step(action)

            score+=r
            state = next_state

        scoreList.append(score)
    
    print("avg_score:", np.mean(scoreList))


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
    epsilon_min = 0.01
    scores = []
    feature_estimate = FeatureEstimate(env, F_idx)
    #while count < 10:

    print("\n============================:: Initialization ::==========================\n")
    randomPolicyFE = randomFE(F_idx)
    print("RandomFE ::")
    print(randomPolicyFE, '\n')

    expertPolicyFE = expertFE(env, trejectories, F_idx)
    print("ExpertFE ::")
    print(expertPolicyFE, '\n')

    expertFE_List = np.matrix([expertPolicyFE])
    agentFE_List = np.matrix([randomPolicyFE])
    
    W,B = QP_optimizer(expertFE_List, agentFE_List)

    while True:
        state = env.reset()
        score = 0


        while True:
            state_idx = idx_to_state(env, state)

            if random.random() < epsilon:
                action = np.random.randint(0, 2)  # random #3
            else:
                action = np.argmax(q_table[state_idx])

            next_state, reward, done, _ = env.step(action)
            next_state_idx = idx_to_state(env, next_state)

            features = feature_estimate.get_features(next_state)
            irl_reward = np.dot(W, features) + B

            # Update Q-table
            q_1 = q_table[state_idx][action]
            q_2 = reward + gamma * max(q_table[next_state_idx])
            q_table[state_idx][action] += q_learning_rate * (q_2 - q_1)

            score += reward
            state = next_state
            epsilon = np.maximum(epsilon_min, 0.99*epsilon)

            if done:
                scores.append(score)
                if np.mean(scores[-min(10, len(scores)):]) > -120:
                    print(":: TOUCH DOWN EPISODE %d / SCORE %d \n" % (episode, score))
                    np.save("best_q_table", arr=q_table)
                    print(":: Complete Q-learning.\n")

                    play(env, q_table)

                    env.close()
                    sys.exit()

                episode += 1
                break

        if episode % 1000 == 0:
            #1000ep마다 score 표시
            print('{} episode | score: {:.1f}'.format(episode, score))

        if episode % 10000 == 0:
            #10000ep마다 weight optimize
            tempFE = calc_FE(env, q_table)
            agentFE_List = add_policyList(agentFE_List, tempFE)
            W,B = QP_optimizer(expertFE_List, agentFE_List)

