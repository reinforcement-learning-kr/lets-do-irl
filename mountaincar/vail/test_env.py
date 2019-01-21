import gym

env = gym.make("MountainCar-v0")

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space

print('state size:', num_inputs)
print('action size:', num_actions)

for episode in range(10000):
    env.reset()

    while True:
        env.render()
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        print("state : {} | action : {} | reward : {}".format(
                state, action, reward))
        
        if done:
            break