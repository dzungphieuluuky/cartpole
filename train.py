import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')
print(f'Observation space: {env.observation_space}')
print(f'Action space: {env.action_space}')

# hyperparameter
learning_rate = 0.1
gamma = 0.95
episodes = 500_000
epsilon_min = 0.01
epislon_decay = 0.01

# discretization
num_bucket = 20
discrete_os = [num_bucket] * len(env.observation_space.low)
print(f"Discrete observation: {discrete_os}")
discrete_os_window = (env.observation_space.high - env.observation_space.low) / discrete_os

def get_discrete_state(state):
    dis = (state - env.observation_space.low) / discrete_os_window
    return tuple(dis.astype(int))

def training(episodes):
    epsilon = 1
    q_table = np.random.uniform(low=-2, high=2, size=discrete_os + [env.action_space.n])
    episode_rewards = []
    episode_store = []
    for episode in range(episodes):
        ep_reward = 0
        episode_store.append(episode)
        discrete_state = get_discrete_state(np.array(env.reset()[0], dtype=object))    
        done = False
        print(f"Episode: {episode}")
        while not done:
            
            if (np.random.randint(0, 1) < epsilon):
                action = np.random.randint(0, env.action_space.n)
            else:
            # get the action index with the most quality
                action = np.argmax(q_table[discrete_state])
            # step the env to get the next state
            next_state, reward, done, truncate, info = env.step(action)
            # discretize the next state
            discrete_next_state = get_discrete_state(next_state)

            # updte the q table
            max_target_q = np.max(q_table[discrete_next_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1 - learning_rate) * current_q + learning_rate * (reward + gamma * max_target_q)
            q_table[discrete_state + (action, )] = new_q

            if done or truncate:
                break

            # update the current state to the next state
            discrete_state = discrete_next_state
            ep_reward += reward
            if epsilon >= epsilon_min:
                epsilon *= (1 - epislon_decay)
            
        print(f'Reward: {ep_reward}')
        episode_rewards.append(ep_reward)
    
    plt.plot(episode_store, episode_rewards, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend(episode_store, episode_rewards)
    plt.show()
    np.save('cartpole_qtable.npy', q_table)

training(episodes)