
import numpy as np

class FeatureEstimate:
    def __init__(self, env, num_features):
        self.env = env
        self.num_features = num_features
        self.feature = np.zeros(self.num_features)

    def gaussian(self, x, mu):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(1., 2.)))

    def get_features(self, state):
        env_low = self.env.observation_space.low
        env_high = self.env.observation_space.high
        env_distance = (env_high - env_low) / (self.num_features - 1)

        for i in range(self.num_features - 1):
            self.feature[i] = self.gaussian(state[0], env_low[0] + i * env_distance[0])

        return self.feature