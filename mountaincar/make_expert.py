import gym
import readchar

# # MACROS
Push_Left = 0
No_Push = 1
Push_Right = 2

# Key mapping
arrow_keys = {
    '\x1b[D': Push_Left,
    '\x1b[B': No_Push,
    '\x1b[C': Push_Right}

env = gym.make('MountainCar-v0')
env.render()
env.reset()

cnt = 0

while True:
    key = readchar.readkey()
    if key not in arrow_keys.keys():
        print("Game aborted!")
        break

    action = arrow_keys[key]
    state, reward, done, _ = env.step(action)
    env.render() 
    print("State: ", state, "Action: ", action,
          "Reward: ", reward, "Iteration Count", cnt)
    cnt += 1

    if done:
        print("Finished with reward", reward)
        break