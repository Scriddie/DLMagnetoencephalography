import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        # self.dropout0 = nn.Dropout(0.2)
        self.cnn1 = nn.Conv1d(input_dim, 16, kernel_size=5, stride=2)
        # self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(7968, output_dim)
        # self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        output = x
        # output = self.dropout0(output)
        output = F.relu(self.cnn1(output))
        # output = torch.squeeze(output)
        output = torch.flatten(output, 1)
        # output = self.dropout1(output)
        output = self.fc1(output)
        # output = self.sigmoid(output)
        # output = self.softmax(output)
        return(output)

