import numpy as np
import torch
import torch.nn as nn
np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cpu")

# Arima baseline?

# sliding window nn
class ANN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # TODO think about how to scale up architecture
        # self.hidden_dims = [10]
        self.fc1 = nn.Linear(input_dim, 10)
        self.fc2 = nn.Linear(10, output_dim)
        # TODO make sure this dim is always right!
        self.softmax = nn.Softmax(dim=1)
        # self.softmax = torch.nn.Softmax(10, output_dim)

    def forward(self, x):
        output = self.fc1(x)
        output = self.fc2(output)
        output = self.softmax(output)
        return(output)

# Conv1dCNN
class CNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.cnn1 = nn.Conv1d(input_dim, 1, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(498, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        output = self.cnn1(x)
        output = torch.squeeze(output)
        output = self.fc1(output)
        output = self.softmax(output)
        return(output)

