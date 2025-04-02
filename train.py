import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from network import Net

BATCH_SIZE = 16
PERCENTILE = 70
    
episode_steps = []

def iterate_batches(env, net, batch_size):
    # create batches and initialize
    batch = []
    episode_reward = 0
    episode_steps = []

    # get the first observation
    obs = np.array(env.reset()[0], dtype=float)

    # define the softmax activation
    softmax = nn.Softmax(1)
    
    while True:
        # transform into torch tensor
        obs_v = torch.FloatTensor(np.array(obs))

        # reshape into [1, obs_v.shape()] 2D array to feed into network
        obs_v = obs_v.reshape(1, -1)
        act_probs_v = softmax(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]

        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, done, truncate, info = env.step(action)
        episode_reward += reward
        episode_steps.append((obs, action))

        if done or truncate:
            batch.append((episode_reward, episode_steps))
            episode_reward = 0
            episode_steps = []

            # get the new first observation after termination
            next_obs = np.array(env.reset()[0], dtype=float)

            # yield batch and continue from this location
            if len(batch) == batch_size:
                yield batch # generator
                batch = []
        
        # state transition
        obs = next_obs

def filter_batch(batch, percentile):
    rewards = list(map(lambda s : s[0], batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for rewards, steps in batch:
        if rewards < reward_bound:
            continue
        train_obs.extend(map(lambda s : s[0], steps))
        train_act.extend(map(lambda s : s[1], steps))
    
    train_obs_tensor = torch.FloatTensor(np.array(train_obs))
    train_act_tensor = torch.LongTensor(np.array(train_act))
    return train_obs_tensor, train_act_tensor, reward_bound, reward_mean

if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.n
    print(obs_size, act_size)

    net = Net(obs_size, act_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    iterations = []
    losses = []

    batches = iterate_batches(env, net, BATCH_SIZE)
    iter = 0
    for batch in batches:
        iter += 1
        obs_sample, act_sample, reward_bound, reward_mean = filter_batch(batch, PERCENTILE)
        act_predicted = net(obs_sample)
        optimizer.zero_grad()
        loss = loss_fn(act_predicted, act_sample)
        loss.backward()
        optimizer.step()
        print(f"Iteration: {iter}, Loss: {loss}, Reward mean: {reward_mean}, Reward bound: {reward_bound}")
        iterations.append(iter)
        losses.append(torch.detach(loss).numpy())
        results = [iterations, losses]

        if reward_mean > 199:
            print("Training successfully!")
            torch.save(net.state_dict(), "model/cartpole.pth")
            np.save('results.npy', results)
            break
