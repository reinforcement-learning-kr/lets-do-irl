import torch
import numpy as np
from utils.utils import *

def train_vdb(vdb, memory, vdb_optim, demonstrations, beta, args):
    memory = np.array(memory) 
    states = np.vstack(memory[:, 0]) 
    actions = list(memory[:, 1]) 

    states = torch.Tensor(states)
    actions = torch.Tensor(actions)

    criterion = torch.nn.BCELoss()

    for _ in range(args.vdb_update_num):
        learner, l_mu, l_logvar = vdb(torch.cat([states, actions], dim=1))
        demonstrations = torch.Tensor(demonstrations)
        expert, e_mu, e_logvar = vdb(demonstrations)

        l_kld = kl_divergence(l_mu, l_logvar)
        l_kld = l_kld.mean()
        
        e_kld = kl_divergence(e_mu, e_logvar)
        e_kld = e_kld.mean()
        
        kld = 0.5 * (l_kld + e_kld)
        bottleneck_loss = kld - args.i_c

        beta = max(0, beta + args.alpha_beta * bottleneck_loss)

        vdb_loss = criterion(learner, torch.ones((states.shape[0], 1))) + \
                    criterion(expert, torch.zeros((demonstrations.shape[0], 1))) + \
                    beta * bottleneck_loss
                
        vdb_optim.zero_grad()
        vdb_loss.backward(retain_graph=True)
        vdb_optim.step()

    expert_acc = ((vdb(demonstrations)[0] < 0.5).float()).mean()
    learner_acc = ((vdb(torch.cat([states, actions], dim=1))[0] > 0.5).float()).mean()

    return expert_acc, learner_acc
    

def train_actor_critic(actor, critic, memory, actor_optim, critic_optim, args):
    memory = np.array(memory) 
    states = np.vstack(memory[:, 0]) 
    actions = list(memory[:, 1]) 
    rewards = list(memory[:, 2]) 
    masks = list(memory[:, 3]) 

    old_values = critic(torch.Tensor(states))
    returns, advants = get_gae(rewards, masks, old_values, args)
    
    mu, std = actor(torch.Tensor(states))
    old_policy = log_prob_density(torch.Tensor(actions), mu, std)

    criterion = torch.nn.MSELoss()
    n = len(states)
    arr = np.arange(n)

    for _ in range(args.ppo_update_num):
        np.random.shuffle(arr)

        for i in range(n // args.batch_size): 
            batch_index = arr[args.batch_size * i : args.batch_size * (i + 1)]
            batch_index = torch.LongTensor(batch_index)
            
            inputs = torch.Tensor(states)[batch_index]
            actions_samples = torch.Tensor(actions)[batch_index]
            returns_samples = returns.unsqueeze(1)[batch_index]
            advants_samples = advants.unsqueeze(1)[batch_index]
            oldvalue_samples = old_values[batch_index].detach()
            
            values = critic(inputs)
            clipped_values = oldvalue_samples + \
                             torch.clamp(values - oldvalue_samples,
                                         -args.clip_param, 
                                         args.clip_param)
            critic_loss1 = criterion(clipped_values, returns_samples)
            critic_loss2 = criterion(values, returns_samples)
            critic_loss = torch.max(critic_loss1, critic_loss2).mean()

            loss, ratio, entropy = surrogate_loss(actor, advants_samples, inputs,
                                         old_policy.detach(), actions_samples,
                                         batch_index)
            clipped_ratio = torch.clamp(ratio,
                                        1.0 - args.clip_param,
                                        1.0 + args.clip_param)
            clipped_loss = clipped_ratio * advants_samples
            actor_loss = -torch.min(loss, clipped_loss).mean()

            loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

            critic_optim.zero_grad()
            loss.backward(retain_graph=True) 
            critic_optim.step()

            actor_optim.zero_grad()
            loss.backward()
            actor_optim.step()

def get_gae(rewards, masks, values, args):
    rewards = torch.Tensor(rewards)
    masks = torch.Tensor(masks)
    returns = torch.zeros_like(rewards)
    advants = torch.zeros_like(rewards)
    
    running_returns = 0
    previous_value = 0
    running_advants = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + (args.gamma * running_returns * masks[t])
        returns[t] = running_returns

        running_delta = rewards[t] + (args.gamma * previous_value * masks[t]) - \
                                        values.data[t]
        previous_value = values.data[t]
        
        running_advants = running_delta + (args.gamma * args.lamda * \
                                            running_advants * masks[t])
        advants[t] = running_advants

    advants = (advants - advants.mean()) / advants.std()
    return returns, advants

def surrogate_loss(actor, advants, states, old_policy, actions, batch_index):
    mu, std = actor(states)
    new_policy = log_prob_density(actions, mu, std)
    old_policy = old_policy[batch_index]

    ratio = torch.exp(new_policy - old_policy)
    surrogate_loss = ratio * advants
    entropy = get_entropy(mu, std)

    return surrogate_loss, ratio, entropy