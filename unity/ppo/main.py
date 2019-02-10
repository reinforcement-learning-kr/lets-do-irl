import os
import platform
import argparse
import numpy as np
from unityagents import UnityEnvironment

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from model import Actor, Critic
from ppo import process_memory, train_model
from utils.memory import Memory
from utils.running_state import ZFilter
from utils.utils import to_tensor, get_action, save_checkpoint

parser = argparse.ArgumentParser(description='Setting for unity walker agent')
parser.add_argument('--render', default=False, action='store_true',
                    help='if you dont want to render, set this to False')
parser.add_argument('--train', default=False, action='store_true',
                    help='if you dont want to train, set this to False')
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--gamma', type=float, default=0.995, 
                    help='discount factor')
parser.add_argument('--lamda', type=float, default=0.95, 
                    help='GAE hyper-parameter')
parser.add_argument('--hidden_size', type=int, default=512,
                    help='hidden unit size of actor and critic networks')
parser.add_argument('--critic_lr', type=float, default=0.0003)
parser.add_argument('--actor_lr', type=float, default=0.0003)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--max_iter', type=int, default=2000000,
                    help='the number of max iteration')
parser.add_argument('--time_horizon', type=int, default=1000,
                    help='the number of time horizon (step number) T ')
parser.add_argument('--l2_rate', type=float, default=0.001,
                    help='l2 regularizer coefficient')
parser.add_argument('--clip_param', type=float, default=0.1,
                    help='hyper parameter for ppo policy loss and value loss')
parser.add_argument('--activation', type=str, default='swish',
                    help='you can choose between tanh and swish')
parser.add_argument('--logdir', type=str, default='logs',
                    help='tensorboardx logs directory')
parser.add_argument('--env', type=str, default='plane',
                    help='environment, plane or curved')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    if platform.system() == 'Darwin':
        env_name = "./env/{}-mac".format(args.env)
    elif platform.system() == 'Linux':
        env_name = "./env/{}-linux/plane-walker".format(args.env)
    elif platform.system() == 'Windows':
        env_name = "./env/{}-win/Unity Environment".format(args.env)

    train_mode = args.train
    torch.manual_seed(500)

    if args.render:
        env = UnityEnvironment(file_name=env_name)
    else:
        env = UnityEnvironment(file_name=env_name, no_graphics=True)

    # setting for unity ml-agent
    default_brain = env.brain_names[0]
    brain = env.brains[default_brain]
    env_info = env.reset(train_mode=train_mode)[default_brain]

    num_inputs = brain.vector_observation_space_size
    num_actions = brain.vector_action_space_size
    num_agent = env._n_agents[default_brain]

    print('state size:', num_inputs)
    print('action size:', num_actions)
    print('agent count:', num_agent)
    
    writer = SummaryWriter(args.logdir)
    # running average of state
    running_state = ZFilter((num_agent,num_inputs), clip=5)

    actor = Actor(num_inputs, num_actions, args).to(device)
    critic = Critic(num_inputs, args).to(device)

    if torch.cuda.is_available():
        actor = actor.cuda()
        critic = critic.cuda()

    if args.load_model is not None:
        saved_ckpt_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model))
        ckpt = torch.load(saved_ckpt_path, map_location='cpu')

        actor.load_state_dict(ckpt['actor'])
        critic.load_state_dict(ckpt['critic'])

        running_state.rs.n = ckpt['z_filter_n']
        running_state.rs.mean = ckpt['z_filter_m']
        running_state.rs.sum_square = ckpt['z_filter_s']

        print("Loaded OK ex. Zfilter N {}".format(running_state.rs.n))

    states = running_state(env_info.vector_observations)

    actor_optim = optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=args.critic_lr,
                              weight_decay=args.l2_rate)

    scores = []
    score_avg = 0

    for iter in range(args.max_iter):
        actor.eval(), critic.eval()
        memory = [Memory() for _ in range(num_agent)]

        steps = 0
        score = 0

        while steps < args.time_horizon:
            steps += 1

            mu, std, _ = actor(to_tensor(states))
            actions = get_action(mu, std)
            env_info = env.step(actions)[default_brain]

            next_states = running_state(env_info.vector_observations)
            rewards = env_info.rewards
            dones = env_info.local_done
            masks = list(~(np.array(dones)))

            for i in range(num_agent):
                memory[i].push(states[i], actions[i], rewards[i], masks[i])

            score += rewards[0]
            states = next_states

            if dones[0]:
                scores.append(score)
                score = 0
                episodes = len(scores)
                if len(scores) % 1 == 0:
                    score_avg = np.mean(scores[-min(10, episodes):])
                    print('{}th episode : last 10 episode mean score of 1st agent is {:.2f}'.format(
                        episodes, score_avg))

        writer.add_scalar('log/score', float(score_avg), iter)
        actor.train(), critic.train()

        sts, ats, returns, advants, old_policy, old_value = [], [], [], [], [], []

        for i in range(num_agent):
            batch = memory[i].sample()
            st, at, rt, adv, old_p, old_v = process_memory(actor, critic, batch, args)
            sts.append(st)
            ats.append(at)
            returns.append(rt)
            advants.append(adv)
            old_policy.append(old_p)
            old_value.append(old_v)

        sts = torch.cat(sts)
        ats = torch.cat(ats)
        returns = torch.cat(returns)
        advants = torch.cat(advants)
        old_policy = torch.cat(old_policy)
        old_value = torch.cat(old_value)

        train_model(actor, critic, actor_optim, critic_optim, sts, ats, returns, advants,
                    old_policy, old_value, args)

        if iter % 100:
            score_avg = int(score_avg)

            model_path = os.path.join(os.getcwd(),'save_model')
            if not os.path.isdir(model_path):
                os.makedirs(model_path)

            ckpt_path = os.path.join(model_path, 'ckpt_'+ str(score_avg)+'.pth.tar')

            save_checkpoint({
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'z_filter_n':running_state.rs.n,
                'z_filter_m': running_state.rs.mean,
                'z_filter_s': running_state.rs.sum_square,
                'args': args,
                'score': score_avg
            }, filename=ckpt_path)

    env.close()
