import preprocessing as pp
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import random_split
from torch.utils.data import TensorDataset, DataLoader
import itertools
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import pandas as pd
import models
from copy import deepcopy
from train_functions import *
from datetime import datetime
import os
np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cpu")

# Conv1dCNN with Batchnorm
class CNNSideways(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dropout0 = nn.Dropout(0.6)
        self.cnn1 = nn.Conv1d(input_dim, 25, kernel_size=5, stride=2)
        self.dropout1 = nn.Dropout(0.5)
        self.cnn2 = nn.Conv1d(498, 10, kernel_size=5, stride=2)
        self.dropout2 = nn.Dropout(0.6)
        self.fc1 = nn.Linear(110, output_dim)
        # self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        output = self.dropout0(x)

        output = F.relu(self.cnn1(output))
        output = output.permute(0, 2, 1)
        output = self.dropout1(output)

        output = F.relu(self.cnn2(output))
        output = torch.flatten(output, 1)
        output = self.dropout2(output)
        # print(output.shape)
        output = self.fc1(output)
        # output = self.sigmoid(output)
        # output = self.softmax(output)
        return(output)


# Conv1dCNN with Batchnorm
class CNNDropout(nn.Module):
    def __init__(self, input_dim, output_dim, 
        droput_visible=0.5, dropout_hidden=0.5, kernel_size=5, stride=2):
        super().__init__()
        # TODO experiment with dropout2d?
        self.dropout1 = nn.Dropout(droput_visible)
        self.cnn1 = nn.Conv1d(input_dim, 25, kernel_size=kernel_size, stride=stride)
        self.dropout2 = nn.Dropout(dropout_hidden)
        self.cnn2 = nn.Conv1d(25, 1, kernel_size=kernel_size, stride=stride)
        self.dropout3 = nn.Dropout(dropout_hidden)
        self.fc1 = nn.Linear(247, output_dim)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        output = self.dropout1(x)

        output = self.cnn1(output)
        output = F.relu(output)
        output = self.dropout2(output)

        output = self.cnn2(output)
        output = F.relu(output)
        output = self.dropout3(output)

        output = torch.squeeze(output)
        output = self.fc1(output)
        # output = self.softmax(output)
        return(output)

# 6 layer, multiple conv, relu after each conv, dropout, batchnorm

# Conv1dCNN with Batchnorm
class CNNBatchNorm(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # TODO check for output dimensions
        # experiment with dropout2d?
        self.cnn1 = nn.Conv1d(input_dim, 25, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm1d(25)
        self.cnn2 = nn.Conv1d(25, 1, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm1d(247)
        self.fc1 = nn.Linear(247, output_dim)
        # self.fc2 = nn.Linear(50, output_dim)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        output = self.cnn1(x)
        output = self.bn1(output)
        output = F.relu(output)
        # print(output.shape)
        output = self.cnn2(output)
        output = torch.squeeze(output)
        output = self.bn2(output)
        output = F.relu(output)

        output = self.fc1(output)
        # output = self.softmax(output)
        return(output)


class CNN2D(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=5, stride=2):
        super().__init__()
        self.cnn1 = nn.Conv2d(1, 20, kernel_size=kernel_size, stride=stride)
        self.cnn2 = nn.Conv2d(20, 20, kernel_size=kernel_size, stride=stride)
        self.cnn3 = nn.Conv2d(20, 20, kernel_size=kernel_size, stride=stride)
        self.cnn4 = nn.Conv2d(20, 1, kernel_size=kernel_size, stride=stride)
        # self.D11 = nn.Conv1d(28, 1, kernel_size=kernel_size, stride=stride)
        self.fc1 = nn.Linear(708, output_dim)
    def forward(self, x):
        x = x.unsqueeze(1)
        # print(x.shape)
        output = F.relu(self.cnn1(x))
        # print(output.shape)
        output = F.relu(self.cnn2(output))
        # print(output.shape)
        output = F.relu(self.cnn3(output))
        # print(output.shape)
        output = F.relu(self.cnn4(output))
        output = output.squeeze()
        # output = output.permute(0, 2, 1)
        # print(output.shape)
        # output = self.D11(output)
        # print(output.shape)
        output = torch.flatten(output, 1)
        # print(output.shape)
        output = self.fc1(output)
        return(output)

# x_tens.shape
# a = nn.Conv1d(248, 1, kernel_size=5, stride=1)
# print(a)
# b = a(x_tens)
# c = torch.squeeze(b)
# cnn = CNN(x_tens.size(1), n_classes)
# cnn(x_tens)

if __name__ == "__main__":
    all_subjects = pp.load_data(type_name="Intra", train_test="train", downsampling=10)
    datasets = []
    label_list = []
    subject, task, mat = next(all_subjects)
    datasets.append(mat)
    label_list.append(task)

    preprocessing_params = {
        "downsampling": 10, 
        "window_size": 1000, 
        "keep_fraction": 0.01, 
        "scale_observations": True
    }

    x_tens, y_tens, label_dict = pp.preprocess(datasets, label_list, **preprocessing_params)

    preparation_params = {
        "train_fraction": 0.8,
        "n_classes": len(np.unique(y_tens)),
        "model_type": "cnn"
    }
    
    train_loader, cv_loader = prepare_data(x_tens, y_tens, **preparation_params)

    x_batch, y_batch = next(iter(train_loader))
    
    # define CNN
    cnn = models.CNN(input_dim=248, output_dim=preparation_params["n_classes"])
    cnn.to(device)
    
    model_params = {
        "loss_fn": torch.nn.BCELoss(),
        "optimizer": torch.optim.Adam(cnn.parameters(), lr=1e-3),
        "early_stopping": 3,
        "model_name": "intra_CNN1"
    }
    pass