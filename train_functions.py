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


def prepare_data(x_tens, y_tens, train_fraction, n_classes, model_type):
    # reshape data
    all_cat_labels = np.zeros((y_tens.size(0), n_classes))
    for i in range(y_tens.size(0)):
        j = y_tens[i].numpy()
        all_cat_labels[i][j] = 1

    if model_type == "ann":
        x = torch.flatten(x_tens, start_dim=1)
    else:
        x = x_tens

    y = torch.Tensor(all_cat_labels)
    dataset = TensorDataset(x, y)
    train_len = int(train_fraction* y.size(0))
    cv_len = y.size(0) - train_len
    train_dataset, cv_dataset = random_split(dataset, [train_len, cv_len])
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    if cv_len > 0:
        cv_loader = DataLoader(dataset=cv_dataset, batch_size=16, shuffle=True)
    else:
        cv_loader = None
    return train_loader, cv_loader


def evaluate_sample(model, cv_loader, label_dict, title, model_path, intra_cross):
    with torch.no_grad():
        # predict on all cv instances
        preds = None
        y = None
        for x_batch, y_batch in cv_loader:
            x_batch, y_batch = next(iter(cv_loader))
            batch_preds = model(x_batch).detach().numpy()
            if preds is None:
                preds = batch_preds
                y = y_batch.numpy()
            else: 
                preds = np.vstack((preds, batch_preds))
                y = np.vstack((y, y_batch))
    y = np.argmax(y, axis=1)
    y_pred = np.argmax(preds, axis=1)

    # accuracy
    acc = accuracy_score(y, y_pred)
    acc_df = pd.DataFrame({
        "accuracy": [acc]
    })
    print(acc_df)
    acc_df.to_latex(open(f"{model_path}/accuracy_table-{title}.txt", "w"), index=False)

    # confusion
    cm = confusion_matrix(y, y_pred)
    names = list(label_dict)
    df = pd.DataFrame(np.round(cm, 4), index=names, columns=names)
    sns.heatmap(df, annot=True, fmt='g')
    plt.title(f"{title} - ConfutionMatrix")
    plt.tight_layout()
    plt.savefig(f"{model_path}/{intra_cross}_{title} - ConfutionMatrix.png", dpi=500)
    plt.show()

