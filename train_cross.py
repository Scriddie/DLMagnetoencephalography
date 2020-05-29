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


def train_piecewise(model, n_epochs, max_batches, model_params, visualize):
    os.mkdir(model_path)
    preprocessing_params = {
        "downsampling": 10, 
        "window_size": 1000, 
        "keep_fraction": 0.01, 
        "scale_observations": True}

    train_losses = []
    cv_losses = []
    best_cv_loss = np.inf
    early_stopping = 0
    best_model = None
    best_model_params = None
    # epochs
    for epoch in range(n_epochs):
        all_tasks = pp.load_data(type_name="Cross", train_test="train", downsampling=preprocessing_params["downsampling"])
        print(f"Starting training epoch {epoch}")
        # load one task for one subject at a time
        total_train_loss = 0.
        total_cv_loss = 0.
        for j, (subject, task, mat) in enumerate(all_tasks):
            # TODO debug on rest only
            # if task != "rest":
            #     continue
            datasets = [mat]
            label_list = [task]

            # prepare data
            x_tens, y_tens, label_dict = pp.preprocess(datasets, label_list, **preprocessing_params)
            preparation_params = {
                "train_fraction": 0.8,
                "n_classes": 4,
                "model_type": "cnn"}
            train_loader, cv_loader = prepare_data(x_tens, y_tens, **preparation_params)

            print(f"\tTraining on {subject}, {task}")
            # batch gradient descent
            for n_batch, (x_batch, y_batch) in enumerate(train_loader):
                if n_batch > max_batches:
                    break
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                model_params["optimizer"].zero_grad()
                output = model.forward(x_batch)
                loss = model_params["loss_fn"](output, y_batch)
                total_train_loss += loss.item() / x_batch.size(0)
                loss.backward()
                model_params["optimizer"].step()

                with torch.no_grad():
                    x_batch_cv, y_batch_cv = next(iter(cv_loader))
                    x_batch_cv = x_batch_cv.to(device)
                    y_batch_cv = y_batch_cv.to(device)
                    cv_preds = model(x_batch_cv)
                    cv_loss = model_params["loss_fn"](cv_preds, y_batch_cv)
                    total_cv_loss += cv_loss.item() / x_batch.size(0)
                
        train_losses.append(total_train_loss)
        cv_losses.append(total_cv_loss)
        print(f"\t\tTrain loss: {total_train_loss:.3f}  |  CV loss: {total_cv_loss:.3f}")
        if cv_losses[-1] < best_cv_loss:
            best_cv_loss = cv_losses[-1]
            best_model_params = deepcopy(model.state_dict())
        if cv_losses[-1] > min(cv_losses):
            early_stopping += 1
        else:
            early_stopping = 0
        if early_stopping > model_params["early_stopping"]:
            print(f"Early stopping after epoch {epoch}.")
            break

    model_name = model_params["model_name"]
    torch.save(best_model_params, f"{model_path}/{model_name}.pt")
    model.load_state_dict(best_model_params)

    plt.plot(range(epoch+1), train_losses)
    plt.plot(range(epoch+1), cv_losses)
    plt.title("Learning Curves")
    if visualize:
        plt.savefig(f"{model_path}/cross_learning_curves.png", dpi=500)
        plt.show()


if __name__ == "__main__":

    # x_tens.shape
    # a = nn.Conv1d(248, 1, kernel_size=5, stride=1)
    # print(a)
    # b = a(x_tens)
    # c = torch.squeeze(b)
    # cnn = CNN(x_tens.size(1), n_classes)
    # cnn(x_tens)

    model_path = "models/cross_" + datetime.now().strftime("%Y-%m-%d_%H:%M")[0:16]
    
    # define CNN for one task as input
    cnn = models.CNN(input_dim=248, output_dim=4)
    cnn.to(device)
    
    datetime.now().strftime("%Y-%m-%d_%H:%s")[0:16]
    model_params = {
        "loss_fn": torch.nn.BCELoss(),
        "optimizer": torch.optim.Adam(cnn.parameters(), lr=1e-3),
        "early_stopping": 3,
        "model_name": "cross_CNN1"
    }

    train_piecewise(cnn, n_epochs=10, max_batches=50, model_params=model_params, visualize=True)

