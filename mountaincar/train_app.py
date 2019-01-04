import numpy as np
import logging
from algorithms.feature_expectation import calcNewFE, randomFE, expertFE
from cvxopt import matrix
from cvxopt import solvers  # convex optimization library
from algorithms.rlstep import q_learning  # get the Reinforcement learner
import gym

FEATURES = 50

# (20, 100, 4)
trejectories = np.load(file="make_expert/expert_trajectories.npy")


class irlAgent:
    def __init__(self, env, randomFE, expertFE, epsilon):
        self.randomPolicy = randomFE
        self.expertPolicy = expertFE

        self.epsilon = epsilon  # termination when t<0.1
        self.randomT = np.linalg.norm(
            np.asarray(self.expertPolicy) - np.asarray(self.randomPolicy))  # norm of the diff in expert and random
        self.policiesFE = {
            self.randomT: self.randomPolicy}  # storing the policies and their respective t values in a dictionary

        print("[ExpertPE - Random atPE] the Start (t) :: \n", self.randomT)
        print("\n==========================================================================\n")

        self.currentT = self.randomT
        self.minimumT = self.randomT
        self.env = env

    def getRLAgentFE(self, W, i):  # get the feature expectations of a new poliicy using RL agent

        Qtable = q_learning(W, i)  # train the agent and save the model in a file used below
        self.qtable = Qtable
        return calcNewFE(self.env, Qtable, W) # return feature expectations by executing the learned policy

    def policyListUpdater(self, W, i):  # add the policyFE list and differences
        tempFE= self.getRLAgentFE(W, i)  # get feature expectations of a new policy respective to the input weights
        hyperDistance = np.abs(np.dot(W, np.asarray(self.expertPolicy) - np.asarray(tempFE)))  # hyperdistance = t
        self.policiesFE[hyperDistance] = tempFE
        return hyperDistance  # t = (weights.tanspose)*(expert-newPolicy)

    def optimalWeightFinder(self):
        f_w = open('weights.txt', 'w')
        f_fe = open('featexp.txt', 'w')
        i = 1
        while True:
            print("\n=============================:: Interation %d ::===========================\n"% (i))
            print("Start QP Solver.")
            W = self.optimization()  # optimize to find new weights in the list of policies
            print("\nWeights ::")
            print(W,'\n')
            f_w.write(str(W))
            f_w.write('\n')
            print("Distances (t) List ::\n", list(self.policiesFE.keys()), '\n')
            self.currentT = self.policyListUpdater(W, i)
            print("Current distance (t) is:: \n", self.currentT, '\n')
            f_fe.write(str(list(self.policiesFE.values())))
            f_fe.write('\n')
            print("===========================================================================\n")
            if self.currentT <= self.epsilon:  # terminate if the point reached close enough
                self.play()
                break
            i += 1
        f_w.close()
        f_fe.close()
        return W

    def optimization(self):  # implement the convex optimization, posed as an SVM problem
        m = len(self.expertPolicy)
        P = matrix(2.0 * np.eye(m), tc='d')  # min ||w||
        q = matrix(np.zeros(m), tc='d')
        policyList = [self.expertPolicy]
        h_list = [1]
        for i in self.policiesFE.keys():
            policyList.append(self.policiesFE[i])
            h_list.append(1)
        policyMat = np.matrix(policyList)
        policyMat[0] = -1 * policyMat[0]
        G = matrix(policyMat, tc='d')
        h = matrix(-np.array(h_list), tc='d')
        sol = solvers.qp(P, q, G, h)

        weights = np.squeeze(np.asarray(sol['x']))
        norm = np.linalg.norm(weights)
        weights = weights / norm
        return weights  # return the normalized weights

    def play(self):

        q_table = self.qtable

        # Create a new game instance.
        env = gym.make('MountainCar-v0')

        state = env.reset()
        score = 0

        while True:
            env.render()
            state_idx = self.idx_to_state(env, state)
            action = np.argmax(q_table[state_idx])
            next_state, _, done, _ = env.step(action)
            state = next_state
            if done:
                break

    def idx_to_state(self, env, state):
        env_low = env.observation_space.low
        env_high = env.observation_space.high
        env_distance = (env_high - env_low) / 50
        position_idx = int((state[0] - env_low[0]) / env_distance[0])
        velocity_idx = int((state[1] - env_low[1]) / env_distance[1])
        state_idx = position_idx + velocity_idx * 50
        return state_idx


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    trejectories = np.load(file="make_expert/expert_trajectories.npy")

    print("\n============================:: Initialization ::==========================\n")
    randomPolicyFE = randomFE(FEATURES)
    print("RandomFE ::")
    print(randomPolicyFE, '\n')
    # ^the random policy feature expectations
    expertPolicyFE = expertFE(env, trejectories, FEATURES)
    print("ExpertFE ::")
    print(expertPolicyFE, '\n')
    # ^feature expectations for the "follow Yellow obstacles" behavior

    epsilon = 0.01
    irlearner = irlAgent(env, randomPolicyFE, expertPolicyFE, epsilon)

    print("\nThe Optimal Weight is ::\n",irlearner.optimalWeightFinder())


