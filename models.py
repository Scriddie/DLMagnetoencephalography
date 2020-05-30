import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cpu")

# Conv1dCNN
class simpleCNN(nn.Module):
    def __init__(self, input_dim, output_dim, n_channels=16):
        super().__init__()
        self.model_name = "ShallowCNN"
        self.cnn1 = nn.Conv1d(input_dim, n_channels, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(7968, output_dim)

    def forward(self, x, dropout=None):
        output = x
        output = F.relu(self.cnn1(output))
        output = torch.flatten(output, 1)
        output = self.fc1(output)
        return(output)

# class simpleCNNDropout(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.model_name = "ShallowCNNDropout"
#         # self.dropout0 = nn.Dropout(0.2)
#         self.cnn1 = nn.Conv1d(input_dim, 16, kernel_size=5, stride=2)
#         self.dropout1 = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(7968, output_dim)

#     def forward(self, x):
#         output = x
#         # output = self.dropout0(output)
#         output = F.relu(self.cnn1(output))
#         output = torch.flatten(output, 1)
#         output = self.dropout1(output)
#         output = self.fc1(output)
#         return(output)

class deepCNN(nn.Module):
    def __init__(self, input_dim, output_dim, 
        droput_visible=0.5, dropout_hidden=0.5, kernel_size=5, stride=2):
        super().__init__()
        self.model_name = "DeepCNN"
        self.cnn1 = nn.Conv1d(input_dim, 128, kernel_size=kernel_size, stride=stride)
        self.cnn2 = nn.Conv1d(128, 64, kernel_size=kernel_size, stride=stride)
        self.cnn3 = nn.Conv1d(64, 32, kernel_size=kernel_size, stride=stride)
        self.fc1 = nn.Linear(3904, 100)
        self.fc2 = nn.Linear(100, output_dim)

    def forward(self, x, dropout=None):
        output = x
        output = self.cnn1(output)
        output = F.relu(output)
        # print(output.shape)
        output = self.cnn2(output)
        output = F.relu(output)
        # print(output.shape)
        output = self.cnn3(output)
        output = F.relu(output)
        output = torch.flatten(output, 1)
        # print(output.shape)
        output = self.fc1(output)
        output = self.fc2(output)
        return(output)

class deepCNNDropout(nn.Module):
    def __init__(self, input_dim, output_dim, 
        droput_visible=0.2, dropout_cv=0.2, dropout_hidden=0.5, kernel_size=5, stride=2):
        super().__init__()
        self.model_name = "DeepCNNDropout"
        self.dropout1 = nn.Dropout(droput_visible)
        self.cnn1 = nn.Conv1d(input_dim, 128, kernel_size=kernel_size, stride=stride)
        self.dropout_cv1 = nn.Dropout(dropout_cv)
        self.cnn2 = nn.Conv1d(128, 64, kernel_size=kernel_size, stride=stride)
        self.dropout_cv2 = nn.Dropout(dropout_cv)
        self.cnn3 = nn.Conv1d(64, 32, kernel_size=kernel_size, stride=stride)
        # self.dropout2 = nn.Dropout(dropout_hidden)
        self.fc1 = nn.Linear(3904, 100)
        self.dropout3 = nn.Dropout(dropout_hidden)
        self.fc2 = nn.Linear(100, output_dim)

    def forward(self, x, dropout=True):
        output = x
        if dropout:
            output = self.dropout1(output)
        output = self.cnn1(output)
        output = F.relu(output)
        if dropout:
            output = self.dropout_cv1(output)
        # print(output.shape)
        output = self.cnn2(output)
        output = F.relu(output)
        if dropout:
            output = self.dropout_cv2(output)
        # print(output.shape)
        output = self.cnn3(output)
        output = F.relu(output)
        output = torch.flatten(output, 1)
        # output = self.dropout2(output)
        # print(output.shape)
        output = self.fc1(output)
        if dropout:
            output = self.dropout3(output)
        output = self.fc2(output)
        return(output)
    

# Conv1dCNN with Batchnorm
class deepCNNBatchNorm(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=5, stride=2):
        super().__init__()
        self.model_name = "DeepCNNBatchNorm"
        self.cnn1 = nn.Conv1d(input_dim, 128, kernel_size=kernel_size, stride=stride)
        self.bn1 = nn.BatchNorm1d(128)
        self.cnn2 = nn.Conv1d(128, 64, kernel_size=kernel_size, stride=stride)
        self.bn2 = nn.BatchNorm1d(64)
        self.cnn3 = nn.Conv1d(64, 32, kernel_size=kernel_size, stride=stride)
        self.fc1 = nn.Linear(3904, 100)
        self.bn3 = nn.BatchNorm1d(100)
        # self.dropout3 = nn.Dropout(dropout_hidden)
        self.fc2 = nn.Linear(100, output_dim)

    def forward(self, x, dropout=None):
        # TODO putting BN before the relu worked great
        output = x
        output = self.cnn1(output)
        output = F.relu(output)
        output = self.bn1(output)
        # print(output.shape)
        output = self.cnn2(output)
        output = F.relu(output)
        output = self.bn2(output)
        # print(output.shape)
        output = self.cnn3(output)
        output = F.relu(output)
        output = torch.flatten(output, 1)
        # print(output.shape)
        output = self.fc1(output)
        output = self.bn3(output)
        # output = self.dropout3(output)
        output = self.fc2(output)
        return(output)


class CNN2D(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=7, stride=3):
        super().__init__()
        self.model_name = "CNN2D"
        self.cnn1 = nn.Conv2d(1, 32, kernel_size=kernel_size, stride=stride)
        self.cnn2 = nn.Conv2d(32, 16, kernel_size=kernel_size, stride=stride)
        self.cnn3 = nn.Conv2d(16, 8, kernel_size=kernel_size, stride=stride)
        # self.D11 = nn.Conv1d(28, 1, kernel_size=kernel_size, stride=stride)
        self.fc1 = nn.Linear(1960, 100)
        self.fc2 = nn.Linear(100, output_dim)
    def forward(self, x, dropout=None):
        x = x.unsqueeze(1)
        # print(x.shape)
        output = F.relu(self.cnn1(x))
        # print(output.shape)
        output = F.relu(self.cnn2(output))
        # print(output.shape)
        output = F.relu(self.cnn3(output))
        # print(output.shape)
        # output = output.permute(0, 2, 1)
        # print(output.shape)
        # output = self.D11(output)
        # print(output.shape)
        output = torch.flatten(output, 1)
        # print(output.shape)
        output = self.fc1(output)
        output = self.fc2(output)
        return(output)


# Conv1dCNN with Batchnorm
class CNNSideways(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model_name = "MultiDirectionConv1D"
        # self.dropout0 = nn.Dropout(0.2)
        self.cnn1 = nn.Conv1d(input_dim, 25, kernel_size=5, stride=2)
        # self.dropout1 = nn.Dropout(0.3)
        self.cnn2 = nn.Conv1d(498, 10, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(110, output_dim)
        # self.dropout2 = nn.Dropout(0.1)
        # self.fc2 = nn.Linear(50, output_dim)
        # self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, dropout=None):
        output = x
        # output = self.dropout0(x)

        output = F.relu(self.cnn1(output))
        output = output.permute(0, 2, 1)
        # output = self.dropout1(output)

        output = F.relu(self.cnn2(output))
        output = torch.flatten(output, 1)
        # print(output.shape)
        output = self.fc1(output)
        # output = self.dropout2(output)
        # output = self.fc2(output)
        # output = self.sigmoid(output)
        # output = self.softmax(output)
        return(output)
