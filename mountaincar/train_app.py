import numpy as np
import logging
from algorithms.feature_expectation import calcNewFE, randomFE, expertFE
from cvxopt import matrix
from cvxopt import solvers  # convex optimization library
from algorithms.rlstep import q_learning  # get the Reinforcement learner
import gym


TRAJECTORIES = 100000  # number of RL training trajectories per iteration of IRL

FEATURES = 50

# (20, 100, 4)
trejectories = np.load(file="make_expert/expert_trajectories.npy")


class irlAgent:
    def __init__(self, env, randomFE, expertFE, epsilon, num_trajectories):
        self.randomPolicy = randomFE
        self.expertPolicy = expertFE
        self.num_trajectories = num_trajectories
        self.epsilon = epsilon  # termination when t<0.1
        self.randomT = np.linalg.norm(
            np.asarray(self.expertPolicy) - np.asarray(self.randomPolicy))  # norm of the diff in expert and random
        self.policiesFE = {
            self.randomT: self.randomPolicy}  # storing the policies and their respective t values in a dictionary
        print("Expert - Random at the Start (t) :: ", self.randomT)
        self.currentT = self.randomT
        self.minimumT = self.randomT
        self.env = env

    def getRLAgentFE(self, W, i):  # get the feature expectations of a new poliicy using RL agent

        Qtable = q_learning(W, i)  # train the agent and save the model in a file used below

        return calcNewFE(self.env, Qtable, W)  # return feature expectations by executing the learned policy

    def policyListUpdater(self, W, i):  # add the policyFE list and differences
        tempFE = self.getRLAgentFE(W, i)  # get feature expectations of a new policy respective to the input weights
        hyperDistance = np.abs(np.dot(W, np.asarray(self.expertPolicy) - np.asarray(tempFE)))  # hyperdistance = t
        self.policiesFE[hyperDistance] = tempFE
        return hyperDistance  # t = (weights.tanspose)*(expert-newPolicy)

    def optimalWeightFinder(self):
        f = open('weights.txt', 'w')
        i = 1
        while True:
            W = self.optimization()  # optimize to find new weights in the list of policies
            print("weights ::", W)
            f.write(str(W))
            f.write('\n')
            print("the distances  ::", self.policiesFE.keys())
            self.currentT = self.policyListUpdater(W, i)
            print("Current distance (t) is:: ", self.currentT)
            if self.currentT <= self.epsilon:  # terminate if the point reached close enough
                break
            i += 1
        f.close()
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


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    env = gym.make('MountainCar-v0')
    trejectories = np.load(file="make_expert/expert_trajectories.npy")

    randomPolicyFE = randomFE(FEATURES)
    print("RandomFE ::", randomPolicyFE)
    # ^the random policy feature expectations
    expertPolicyFE = expertFE(env, trejectories, FEATURES)
    print("ExpertFE ::",expertPolicyFE)
    # ^feature expectations for the "follow Yellow obstacles" behavior

    epsilon = 0.1
    irlearner = irlAgent(env, randomPolicyFE, expertPolicyFE, epsilon, TRAJECTORIES)
    print("The Optimal Weight is ::",irlearner.optimalWeightFinder())
