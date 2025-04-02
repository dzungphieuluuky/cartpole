import gymnasium as gym
import numpy as np
import train
import torch

# initialize the gym environment
env = gym.make('CartPole-v1', render_mode='human')

def play():
    done = False
    total_reward = 0

    # get observation size and action size
    observation_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # load the network
    mynet = train.Net(observation_size, action_size)
    mynet.load_state_dict(torch.load("model/cartpole.pth", weights_only=True))

    state = np.array(env.reset()[0], dtype=float)
    while not done:
        state = torch.FloatTensor(state)
        actions = mynet(state)
        action = torch.argmax(actions).item()
        next_state, reward, done, truncate, info = env.step(action)
        if done or truncate:
            break
        total_reward += reward
        state = next_state
    
    print(f'Total Reward: {total_reward}')

if __name__ == "__main__":
    play()