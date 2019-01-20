import os
import gym
import torch
import argparse

from model import Actor, Critic
from utils.utils import get_action
from utils.running_state import ZFilter

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default="Hopper-v2",
                    help='name of Mujoco environement')
parser.add_argument('--iter', type=int, default=5,
                    help='number of episodes to play')
parser.add_argument("--load_model", type=str, default='ppo_max.tar',
                     help="if you test pretrained file, write filename in save_model folder")

args = parser.parse_args()


if __name__ == "__main__":
    env = gym.make(args.env)
    env.seed(500)
    torch.manual_seed(500)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    print("state size: ", num_inputs)
    print("action size: ", num_actions)

    actor = Actor(num_inputs, num_actions)
    critic = Critic(num_inputs)

    running_state = ZFilter((num_inputs,), clip=5)
    
    if args.load_model is not None:
        pretrained_model_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model))

        pretrained_model = torch.load(pretrained_model_path)

        actor.load_state_dict(pretrained_model['actor'])
        critic.load_state_dict(pretrained_model['critic'])

        running_state.rs.n = pretrained_model['z_filter_n']
        running_state.rs.mean = pretrained_model['z_filter_m']
        running_state.rs.sum_square = pretrained_model['z_filter_s']

        print("Loaded OK ex. ZFilter N {}".format(running_state.rs.n))

    else:
        assert("Should write pretrained filename in save_model folder. ex) python3 test_algo.py --load_model ppo_max.tar")


    actor.eval(), critic.eval()
    for episode in range(args.iter):
        state = env.reset()
        steps = 0
        score = 0
        for _ in range(10000):
            env.render()
            mu, std, _ = actor(torch.Tensor(state).unsqueeze(0))
            action = get_action(mu, std)[0]

            next_state, reward, done, _ = env.step(action)
            next_state = running_state(next_state)
            
            state = next_state
            score += reward
            
            if done:
                print("{} cumulative reward: {}".format(episode, score))
                break
