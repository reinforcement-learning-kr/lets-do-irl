import math
import torch

def get_action(mu, std):
    action = torch.normal(mu, std)
    action = action.data.numpy()
    return action

def log_prob_density(x, mu, std):
    log_prob_density = -(x - mu).pow(2) / (2 * std.pow(2)) \
                     - 0.5 * math.log(2 * math.pi)
    return log_prob_density.sum(1, keepdim=True)

def save_checkpoint(state, filename):
    torch.save(state, filename)