import torch
import numpy as np

def train_model(actor, critic, transitions, actor_optim, critic_optim, args):
    states = torch.stack(transitions.state)
    actions = torch.LongTensor(transitions.action)
    rewards = torch.Tensor(transitions.reward)
    masks = torch.Tensor(transitions.mask)

    # ----------------------------
    # step 1: get returns and GAEs and log probability of old policy
    old_values = critic(states)
    returns, advants = get_gae(rewards, masks, old_values, args)

    policies = actor(states)
    old_policy = policies[range(len(actions)), actions]
    
    criterion = torch.nn.MSELoss()
    n = len(states)
    arr = np.arange(n)
    
    # ----------------------------
    # step 2: get value loss and actor loss and update actor & critic
    # batch를 random suffling하고 mini batch를 추출
    for _ in range(10):
        np.random.shuffle(arr)
        
        for i in range(n // args.batch_size): 
            batch_index = arr[args.batch_size * i : args.batch_size * (i + 1)]
            batch_index = torch.LongTensor(batch_index)
            
            inputs = states[batch_index]
            actions_samples = actions[batch_index]
            returns_samples = returns.unsqueeze(1)[batch_index]
            advants_samples = advants.unsqueeze(1)[batch_index]
            oldvalue_samples = old_values[batch_index].detach()

            values = critic(inputs)
            # clipping을 사용하여 critic loss 구하기 
            clipped_values = oldvalue_samples + \
                             torch.clamp(values - oldvalue_samples,
                                         -args.clip_param, # 0.2
                                         args.clip_param)
            critic_loss1 = criterion(clipped_values, returns_samples)
            critic_loss2 = criterion(values, returns_samples)
            critic_loss = torch.max(critic_loss1, critic_loss2).mean()

            # 논문에서 수식 6. surrogate loss 구하기
            loss, ratio = surrogate_loss(actor, advants_samples, inputs,
                                         old_policy.detach(), actions_samples,
                                         batch_index)

            # 논문에서 수식 7. surrogate loss를 clipping해서 actor loss 만들기
            clipped_ratio = torch.clamp(ratio,
                                        1.0 - args.clip_param,
                                        1.0 + args.clip_param)
            clipped_loss = clipped_ratio * advants_samples
            actor_loss = -torch.min(loss, clipped_loss).mean()

            loss = actor_loss + 0.5 * critic_loss

            critic_optim.zero_grad()
            loss.backward(retain_graph=True) 
            critic_optim.step()

            actor_optim.zero_grad()
            loss.backward()
            actor_optim.step()


def get_gae(rewards, masks, values, args):
    returns = torch.zeros_like(rewards)
    advants = torch.zeros_like(rewards)
    
    running_returns = 0
    previous_value = 0
    running_advants = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + (args.gamma * running_returns * masks[t])
        returns[t] = running_returns

        # 논문에서 수식 10
        running_delta = rewards[t] + (args.gamma * previous_value * masks[t]) - \
                                        values.data[t]
        previous_value = values.data[t]
        
        # 논문에서 수식 14 + lambda 추가
        running_advants = running_delta + (args.gamma * args.lamda * \
                                            running_advants * masks[t])
        advants[t] = running_advants

    advants = (advants - advants.mean()) / advants.std()
    return returns, advants


def surrogate_loss(actor, advants, states, old_policy, actions, batch_index):
    policies = actor(states)
    new_policy = policies[range(len(actions)), actions]
    old_policy = old_policy[batch_index]

    ratio = torch.exp(new_policy - old_policy)
    surrogate_loss = ratio * advants
    return surrogate_loss, ratio