import torch
import torch.nn as nn
import torch.optim as optim

HIDDEN_LAYER = 128
class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_LAYER),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER, output_dim)
        )
    def forward(self, x):
        return self.layer(x)