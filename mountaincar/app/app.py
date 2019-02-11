import numpy as np
import cvxpy as cp
from train import idx_to_state

class FeatureEstimate:
    def __init__(self, env, feature_num):
        self.env = env
        self.feature_num = feature_num
        self.feature = np.ones(self.feature_num)

    def gaussian(self, x, mu):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(1., 2.)))

    def get_features(self, state):
        env_low = self.env.observation_space.low
        env_high = self.env.observation_space.high
        env_distance = (env_high - env_low) / (self.feature_num - 1)

        for i in range(int(self.feature_num/2)):
            # position
            self.feature[i] = self.gaussian(state[0], env_low[0] + i * env_distance[0])
            
            # velocity
            self.feature[i+int(self.feature_num/2)] = self.gaussian(state[1], env_low[1] + i * env_distance[1])

        return self.feature


def random_feature_expectation(feature_num):
    return np.random.normal(size=feature_num)

def expert_feature_expectation(feature_num, gamma, demonstrations, env):
    feature_estimate = FeatureEstimate(env, feature_num)
    feature_expectations = np.zeros(feature_num)
    
    for demo_num in range(len(demonstrations)):
        for demo_length in range(len(demonstrations[0])):
            state = demonstrations[demo_num][demo_length]
            features = feature_estimate.get_features(state)
            feature_expectations += (gamma**(demo_length)) * np.array(features)
    
    feature_expectations = feature_expectations / len(demonstrations)
    
    return feature_expectations


def QP_optimizer(feature_num, learner, expert):
    w = cp.Variable(feature_num)
    
    obj_func = cp.Minimize(cp.norm(w))
    constraints = [(expert-learner) * w >= 2] 

    prob = cp.Problem(obj_func, constraints)
    prob.solve()

    if prob.status == "optimal":
        print("status:", prob.status)
        print("optimal value", prob.value)
    
        weights = np.squeeze(np.asarray(w.value))
        return weights, prob.status
    else:
        print("status:", prob.status)
        
        weights = np.zeros(feature_num)
        return weights, prob.status

def calc_feature_expectation(feature_num, gamma, q_table, env):
    feature_estimate = FeatureEstimate(env, feature_num)
    feature_expectations = np.zeros(feature_num)
    
    scores = []
    demo_num = 20
    
    for _ in range(demo_num):
        state = env.reset()
        demo_length = 0
        score = 0
        done = False
        
        while not done:
            demo_length += 1

            state_idx = idx_to_state(env, state)
            action = np.argmax(q_table[state_idx])
            next_state, reward, done, _ = env.step(action)
            
            features = feature_estimate.get_features(next_state)
            feature_expectations += (gamma**(demo_length)) * np.array(features)

            score += reward
            state = next_state

        scores.append(score)
    
    feature_expectations = feature_expectations/ demo_num
    print("avg_score:", np.mean(scores))
    print("\n")
    print("current_feature_expectation:", feature_expectations)

    return feature_expectations


def add_feature_expectation(learner, temp_learner):
    # agent의 RL과정이 끝난 후 나온 FE를 list에 저장
    learner = np.vstack([learner, temp_learner])
    return learner

def subtract_feature_expectation(learner):
    # infeasible할 시에 맨 위 FE를 list에 제외
    learner = learner[1:][:]
    return learner