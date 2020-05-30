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

if __name__ == "__main__":

    ### evaluate performance on test set
    # TODO: do this for every test set!

    version = "cross_2020-05-29_23:15"
    model_name = "cross_CNN"
    model_path = "models/" + version
    # define CNN for one task as input
    cnn = torch.load(f"{model_path}/"+model_name+".pt")
    cnn.to(device)
    # TODO write all the accuracies to the same latex table
    all_accs = pd.DataFrame()
    for i in range(1, 4):
        preprocessing_params = {
            "downsampling": 10, 
            "window_size": 1000, 
            "keep_fraction": 0.01, 
            "scale_observations": True}
        test_subjects = pp.load_data(type_name="Cross", train_test=f"test{i}", downsampling=preprocessing_params["downsampling"])
        x_test, y_test, test_label_dict = pp.preprocess(test_subjects, **preprocessing_params)
        preparation_params = {
            "train_fraction": 1.,
            "batch_size": 16,
            "n_classes": 4,
            "model_type": "cnn"
        }
        test_loader, _ = prepare_data(x_test, y_test, **preparation_params)
        acc_df = evaluate_sample(cnn, test_loader, test_label_dict, model_path=model_path, title=f"Cross Subject - test{i}", intra_cross="cross")
        acc_df["subject"] = [f"test{i}"]
        all_accs = pd.concat((all_accs, acc_df), axis=0)
    all_accs.to_latex(open(f"{model_path}/accuracy_table.txt", "w"), index=False)
