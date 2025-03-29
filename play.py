import gym
import numpy as np

env = gym.make('CartPole-v1', render_mode='human')
num_bucket = 20
discrete_os = [num_bucket] * len(env.observation_space.low)
discrete_os_window = (env.observation_space.high - env.observation_space.low) / discrete_os

def play(times):
    q_table = np.load('cartpole_qtable.npy')
    performance = 0
    for time in range(times):
        state = np.array(env.reset()[0], dtype=object)

        def get_discrete_state(state):
            dis = (state - env.observation_space.low) / discrete_os_window
            return tuple(dis.astype(int))
        
        done = False
        total_reward = 0
        while not done:
            dis_state = get_discrete_state(state)
            action = np.argmax(q_table[dis_state])
            next_state, reward, done, truncate, info = env.step(action)
            if done or truncate:
                break

            total_reward += reward
            state = next_state
        
        print(f'Total Reward: {total_reward}')
        performance += total_reward
    return performance/times

performace = play(100)
print(f"Performance: {performace}")