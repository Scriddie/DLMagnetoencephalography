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
np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cpu")


def prepare_data(x_tens, y_tens, train_fraction, n_classes, model_type, batch_size):
    # reshape data
    # all_cat_labels = np.zeros((y_tens.size(0), n_classes))
    # for i in range(y_tens.size(0)):
    #     j = y_tens[i].numpy()
    #     all_cat_labels[i][j] = 1

    # if model_type == "ann":
    #     x = torch.flatten(x_tens, start_dim=1)
    # else:
    #     x = x_tens

    # y = torch.Tensor(all_cat_labels)
    dataset = TensorDataset(x_tens, y_tens)
    train_len = int(train_fraction * y_tens.size(0))
    cv_len = y_tens.size(0) - train_len
    train_dataset, cv_dataset = random_split(dataset, [train_len, cv_len])
    if train_len > 0:
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    else:
        train_loader = None
    if cv_len > 0:
        cv_loader = DataLoader(dataset=cv_dataset, batch_size=batch_size, shuffle=True)
    else:
        cv_loader = None
    return train_loader, cv_loader

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = -1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = -1)    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    return acc

def evaluate_sample(model, cv_loader, label_dict, title, model_path, intra_cross, visualize=False):
    with torch.no_grad():
        # predict on all cv instances
        preds = None
        y = None
        for n_batches, (x_batch, y_batch) in enumerate(cv_loader):
            batch_preds = torch.log_softmax(model(x_batch, dropout=False), dim = 1)
            _, batch_preds = torch.max(batch_preds, dim = 1)
            if preds is None:
                preds = batch_preds.detach().numpy()
                y = y_batch.numpy()
            else: 
                preds = np.hstack((preds, batch_preds.detach().numpy()))
                y = np.hstack((y, y_batch))
    
    correct_pred = (y == preds)
    acc = correct_pred.sum() / len(correct_pred)
    # accuracy
    acc_df = pd.DataFrame({
        "accuracy": [acc]
    })
    print(acc_df)

    # n_classes = len(np.unique(y))
    # y_ohe = np.zeros((y.shape[0], n_classes))
    # y_pred_ohe = np.zeros(y_ohe.shape)
    # for i, v in enumerate(y):
    #     y_ohe[i][v] = 1
    #     y_pred_ohe[i][preds[i]] = 1
    cm = confusion_matrix(y, preds)
    names = list(label_dict)
    df = pd.DataFrame(np.round(cm, 4), index=names, columns=names)
    plt.close("all")
    sns.heatmap(df, annot=True, fmt='g', cmap="coolwarm")
    plt.title(f"{title} - ConfusionMatrix")
    plt.tight_layout()
    plt.savefig(f"{model_path}/{intra_cross}_{title} - ConfusionMatrix.png", dpi=500)
    if visualize:
        plt.show()

    return acc_df

