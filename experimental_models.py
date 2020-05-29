import preprocessing as pp
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
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
class CNNDropout(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # TODO check for output dimensions
        # experiment with dropout2d?
        self.cnn1 = nn.Conv1d(input_dim, 1, kernel_size=5, stride=2)
        # self.maxpool
        self.dropout1 = nn.Dropout
        self.fc1 = nn.Linear(498, 50)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        output = self.cnn1(x)
        output = torch.squeeze(output)
        output = self.fc1(output)
        output = self.softmax(output)
        return(output)


# GRU lstm?

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