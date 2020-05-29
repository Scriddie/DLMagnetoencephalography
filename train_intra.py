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
np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cpu")

def train_model(model, n_epochs, train_loader, cv_loader, model_params, visualize):
    # TODO keep and save best model
    train_losses = []
    cv_losses = []
    best_cv_loss = np.inf
    early_stopping = 0
    best_model = None
    best_model_params = None
    for epoch in range(n_epochs):
        total_train_loss = 0.
        total_cv_loss = 0.
        
        # batch gradient descent
        for x_batch, y_batch in train_loader:
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
        print(f"\tEpoch {epoch}\t\tTrain loss: {total_train_loss:.3f}  |  CV loss: {total_cv_loss:.3f}")
        if cv_losses[-1] < best_cv_loss:
            best_cv_loss = cv_losses[-1]
            best_model_params = deepcopy(model.state_dict)
        if cv_losses[-1] > min(cv_losses):
            early_stopping += 1
        else:
            early_stopping = 0
        if early_stopping > model_params["early_stopping"]:
            print(f"Early stopping after epoch {epoch}.")
            break

    # TODO save model
    model_name = model_params["model_name"]
    torch.save(best_model_params, f"models/{model_name}.pt")
    model.state_dict = best_model_params

    plt.plot(range(epoch+1), train_losses)
    plt.plot(range(epoch+1), cv_losses)
    plt.title("Learning Curves")
    if visualize:
        plt.savefig("figures/learning_curves.png", dpi=500)
        plt.show()


if __name__ == "__main__":
    pass
    ### model training
    all_subjects = pp.load_data(type_name="Intra", train_test="train", downsampling=10)

    datasets = []
    label_list = []
    for subject, task, mat in all_subjects:
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
    
    # define CNN
    cnn = models.CNN(input_dim=248, output_dim=preparation_params["n_classes"])
    cnn.to(device)
    
    model_params = {
        "loss_fn": torch.nn.BCELoss(),
        "optimizer": torch.optim.Adam(cnn.parameters(), lr=1e-3),
        "early_stopping": 3,
        "model_name": "intra_CNN1"
    }

    train_model(cnn, 30, train_loader, cv_loader, model_params=model_params)
    evaluate(cnn, cv_loader, label_dict)

    ### evaluate on test set
    cnn.load_state_dict(torch.load("models/"+model_params["model_name"]+".pt"))
    test_subjects = pp.load_data(type_name="Intra", train_test="test")
    test_label_list = list(test_subjects["105923"].keys())
    test_datasets = [test_subjects["105923"][i] for i in test_label_list]
    x_test, y_test, test_label_dict = pp.preprocess(test_datasets, test_label_list, **preprocessing_params)
    preparation_params = {
        "train_fraction": 1,
        "n_classes": len(np.unique(y_test)),
        "model_type": "cnn"
    }
    test_loader, _ = prepare_data(**preparation_params)
    evaluate(cnn, test_loader, test_label_dict, title="Intra Subject", intra_cross="intra")