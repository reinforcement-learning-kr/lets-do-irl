import numpy as np
import gym
import random
import sys
import cvxpy as cp

N_idx = 20
F_idx = 2
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
            #position
            self.feature[i] = self.gaussian(state[0], env_low[0] + i * env_distance[0])
            #velocity
            self.feature[i+int(self.num_features/2)] = self.gaussian(state[1], env_low[1] + i * env_distance[1])

        return self.feature

def random_feature_expectation(num_features):
    return np.random.normal(size=num_features)

def expert_feature_expectation(env, trajectories, num_features):
    featureExpectations = np.zeros(num_features)
    feature_estimate = FeatureEstimate(env, num_features)
    for m in range(len(trajectories)):
        for car_steps in range(len(trajectories[0])):
            state = trajectories[m][car_steps]
            features = feature_estimate.get_features(state)
            featureExpectations += (GAMMA**(car_steps))*np.array(features)
    featureExpectations = featureExpectations / len(trajectories)
    return featureExpectations

def QP_optimizer(expertList, agentList):
    w=cp.Variable(F_idx)
    
    policyMat = agentList
    ExpertMat = expertList
        
    constraints = [(ExpertMat-policyMat)*w >= 2]
        
    obj = cp.Minimize(cp.norm(w))
    prob = cp.Problem(obj, constraints)

    prob.solve()
    print("status:", prob.status)
    print("optimal value", prob.value)
    
    weights = np.squeeze(np.asarray(w.value))
    return weights

def add_policyList(policyList, feature_expectation):
    #agent의 RL과정이 끝난 후 나온 FE를 list에 저장
    policyList = np.vstack([policyList, feature_expectation])
    return policyList

def calc_feature_expectation(env, qtable):
    featureExpectations = np.zeros(F_idx)
    feature_estimate = FeatureEstimate(env, F_idx)
    scoreList = []
    n_traj = 20
    
    for _ in range(n_traj):
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
    
    featureExpectations = featureExpectations/ n_traj
    print("avg_score:", np.mean(scoreList))
    print("\n")
    print("current_feature_expectation:", featureExpectations)

    return featureExpectations


if __name__ == '__main__':
    print(":: Start Q-learning.\n")
    gamma = 0.99
    q_learning_rate = 0.03

    n_states = N_idx**2  # position - 20, velocity - 20
    n_actions = 3
    q_table = np.zeros((n_states, n_actions))  # (400, 3)

    # Create a new game instance.
    env = gym.make('MountainCar-v0')
    episode = 0
    scores = []
    feature_estimate = FeatureEstimate(env, F_idx)
    
    #초기화

    print("\n============================:: Initialization ::==========================\n")
    randomPolicy_feature_expectation = random_feature_expectation(F_idx)
    print("Random_feature_expectation ::")
    print(randomPolicy_feature_expectation, '\n')

    expertPolicy_feature_expectation = expert_feature_expectation(env, trejectories, F_idx)
    print("Expert_feature_expectation ::")
    print(expertPolicy_feature_expectation, '\n')

    expert_feature_expectation_List = np.matrix([expertPolicy_feature_expectation])
    agent_feature_expectation_List = np.matrix([randomPolicy_feature_expectation])
    
    W = QP_optimizer(expert_feature_expectation_List, agent_feature_expectation_List)

    while True:
        state = env.reset()
        score = 0

        while True:
            state_idx = idx_to_state(env, state)

            action = np.argmax(q_table[state_idx])

            next_state, reward, done, _ = env.step(action)
            next_state_idx = idx_to_state(env, next_state)

            features = feature_estimate.get_features(next_state)
            irl_reward = np.dot(W, features)

            # Update Q-table
            q_1 = q_table[state_idx][action]
            q_2 = reward + gamma * max(q_table[next_state_idx])
            q_table[state_idx][action] += q_learning_rate * (q_2 - q_1)

            score += reward
            state = next_state

            if done:
                scores.append(score)
                if np.mean(scores[-min(10, len(scores)):]) > -120:
                    print(":: TOUCH DOWN EPISODE %d / SCORE %d \n" % (episode, np.mean(scores[-min(10, len(scores)):])))
                    np.save("best_q_table", arr=q_table)
                    print(":: Complete Q-learning.\n")
                    
                    env.close()
                    sys.exit()

                episode += 1
                break

        if episode % 1000 == 0:
            #1000ep마다 score 표시
            print('{} episode | score: {:.1f}'.format(episode, np.mean(scores[-1000:])))

        if episode % 10000 == 0:
            #10000ep마다 weight optimize
            temp_feature_expectation = calc_feature_expectation(env, q_table)
            agent_feature_expectation_List = add_policyList(agent_feature_expectation_List, temp_feature_expectation)
            W = QP_optimizer(expert_feature_expectation_List, agent_feature_expectation_List)

