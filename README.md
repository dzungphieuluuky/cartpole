# 🏆 RL Agent for CartPole-v1

## 🎯 Overview
This project implements a **Reinforcement Learning (DRL) agent** using **Cross-entropy method** to solve **CartPole-v1** from Gymnasium.

The goal of the agent is to balance the pole as long as possible. A termination state occurs when the pole is displaced from the vertical line at a certain angle.

## 🚀 Key Feature
- Train using Cross-entropy methods.
- Using prioritized mini batches with rewards at 70% percentile.
- Feed forward neural network.

## 🎖️ Architecture
- Environment: CartPole-v1.
- Loss function: Cross Entropy Loss.
- Network: Feed Forward Neural Network.
- Optimizer: Adam.
- Prioritized mini batches.

## 🌹 Dependencies (Windows)
```bash=
pip install -r requirements.txt
```

## 🐼 References
**Deep Reinforcement Learning Hands-On** by Maxim Lapan.

## 🐧 Usage
- Run `train.py` to train the agent.
- Results will automatically be saved in `results.npy`.
- Model file is saved in `model` folder.
- Run `play.py` to watch the agent play.
- Run `plot_results.py` to watch the training progress graph.
