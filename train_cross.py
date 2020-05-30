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
import experimental_models
from importlib import reload
np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cpu")



def train_model(model, train_loader, cv_loader, n_epochs, max_batches, model_params,
    preprocessing_params, model_path, visualize):
    train_losses = []
    cv_losses = []
    best_cv_loss = np.inf
    best_cv_acc = 0
    early_stopping = 0
    best_state_dict = None
    for epoch in range(n_epochs):
        total_train_loss = 0.
        train_acc = 0.
        total_cv_loss = 0.
        cv_acc = 0.
        # batch gradient descent
        for n_batch, (x_batch, y_batch) in enumerate(train_loader):
            if n_batch > max_batches:
                break
            if y_batch.size(0) == 1:
                continue
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            model_params["optimizer"].zero_grad()
            output = model.forward(x_batch)
            train_acc += multi_acc(output, y_batch)
            loss = model_params["loss_fn"](output, y_batch.long())
            total_train_loss += loss.item() / x_batch.size(0)
            loss.backward()
            model_params["optimizer"].step()

            with torch.no_grad():
                x_batch_cv, y_batch_cv = next(iter(cv_loader))
                x_batch_cv = x_batch_cv.to(device)
                y_batch_cv = y_batch_cv.to(device)
                cv_preds = model(x_batch_cv)
                cv_loss = model_params["loss_fn"](cv_preds, y_batch_cv.long())
                total_cv_loss += cv_loss.item() / x_batch.size(0)
                cv_acc += multi_acc(cv_preds, y_batch_cv)
                
        train_losses.append(total_train_loss)
        cv_losses.append(total_cv_loss)
        epoch_train_acc = train_acc / n_batch
        epoch_cv_acc = cv_acc / n_batch
        print(f"\tEpoch: {epoch}\tTrain loss: {total_train_loss:.3f}; {epoch_train_acc*100:.2f}%  |  CV loss: {total_cv_loss:.3f}; {epoch_cv_acc*100:.2f}%")
        if cv_losses[-1] < best_cv_loss:
            best_cv_loss = cv_losses[-1]
            best_cv_acc = epoch_cv_acc
            best_state_dict = deepcopy(model.state_dict())
        if cv_losses[-1] > min(cv_losses):
            early_stopping += 1
        else:
            early_stopping = 0
        if early_stopping > model_params["early_stopping"]:
            print(f"Early stopping after epoch {epoch}.")
            break

    model_name = model_params["model_name"]
    model.load_state_dict(best_state_dict)
    torch.save(model, f"{model_path}/{model_name}.pt")
    plt.close("all")
    plt.plot(range(epoch+1), train_losses)
    plt.plot(range(epoch+1), cv_losses)
    plt.legend(["Training", "Cross-Validation"])
    plt.title("Learning Curves")
    plt.savefig(f"{model_path}/cross_learning_curves.png", dpi=500)
    if visualize:
        plt.show()
    return best_cv_loss


def lr_grid_search(model_df, test_loaders):
    # TODO grid search learning rate
    learning_rates = np.linspace(1e-4, 1e-3, 5)
    for lr in learning_rates:
        print(f"Learning Rate: {lr}")
        model_path = "models/grid_search/cross_" + datetime.now().strftime("%Y-%m-%d_%H:%M")[0:16]
        os.mkdir(model_path)
        model = experimental_models.CNNDropout(input_dim=248, output_dim=4)
        model.to(device)
        model_params = {
            "loss_fn": torch.nn.CrossEntropyLoss(),
            "optimizer": torch.optim.Adam(model.parameters(), lr=lr),
            "early_stopping": 5,
            "model_name": "CNNDropout"
        }
        best_cv_loss = train_model(model, train_loader, cv_loader, n_epochs=200, max_batches=9999, model_params=model_params, 
            preprocessing_params=preprocessing_params, model_path=model_path, visualize=False)
        # TODO get model train & cv acc
        model_df["Learning Rate"] = [lr]
        model_df["Cross-Validation Loss"] = [best_cv_loss]
        model_df["Model Type"] = [model_params["model_name"]]
        model_df.to_latex(open(f"{model_path}/model_config.txt", "w"), index=False)
        evaluate(model, model_path, test_loaders)

def conv_grid_search(odel_df, test_loaders):
    pass



def evaluate(model, model_path, test_loaders, visualize=False):
    all_accs = pd.DataFrame()
    for i in range(1, 4):
        acc_df = evaluate_sample(model, test_loaders[i-1], test_label_dict, model_path=model_path,
             title=f"Cross Subject - test{i}", intra_cross="cross", visualize=visualize)
        acc_df["subject"] = [f"test{i}"]
        all_accs = pd.concat((all_accs, acc_df), axis=0)
    all_accs.to_latex(open(f"{model_path}/accuracy_table.txt", "w"), index=False)


if __name__ == "__main__":

    model_path = "models/cross_" + datetime.now().strftime("%Y-%m-%d_%H:%M")[0:16]
    os.mkdir(model_path)

    ### load data
    preprocessing_params = {
        "downsampling": 10, 
        "window_size": 1000, 
        "keep_fraction": 0.01, 
        "scale_observations": True}
    all_tasks = pp.load_data(type_name="Cross", train_test="train", downsampling=preprocessing_params["downsampling"])
    x_tens, y_tens, label_dict = pp.preprocess(all_tasks, **preprocessing_params)

    preparation_params = {
        "train_fraction": 0.8,
        "batch_size": 16,
        "n_classes": 4,
        "model_type": "cnn"}
    train_loader, within_cv_loader = prepare_data(x_tens, y_tens, **preparation_params)
    cv_tasks = pp.load_data(type_name="Cross", train_test="CV", downsampling=preprocessing_params["downsampling"])
    cv_x_tens, cv_y_tens, cv_label_dict = pp.preprocess(cv_tasks, **preprocessing_params)
    cv_preparation_params = {
        "train_fraction": 0,
        "batch_size": 16,
        "n_classes": 4,
        "model_type": "cnn"}
    _, cv_loader = prepare_data(cv_x_tens, cv_y_tens, **cv_preparation_params)

    test_loaders = []
    for i in range(1, 4):
        test_subjects = pp.load_data(type_name="Cross", train_test=f"test1", downsampling=preprocessing_params["downsampling"])
        x_test, y_test, test_label_dict = pp.preprocess(test_subjects, **preprocessing_params)
        tl, _ = prepare_data(x_test, y_test, **preparation_params)
        test_loaders.append(tl)

    ### train cnn
    # cnn = models.CNN(input_dim=248, output_dim=4)
    reload(experimental_models)
    reload(models)
    # cnn = experimental_models.CNNDropout(input_dim=248, output_dim=4)
    # cnn = models.CNN(input_dim=248, output_dim=4)
    # cnn = experimental_models.CNN2D(248, 4)
    cnn = experimental_models.CNNSideways(248, 4)
    cnn.to(device)
    
    datetime.now().strftime("%Y-%m-%d_%H:%s")[0:16]
    model_params = {
        "loss_fn": torch.nn.CrossEntropyLoss(),
        "optimizer": torch.optim.Adam(cnn.parameters(), lr=5e-5),
        "early_stopping": 5,
        "model_name": "cross_CNN"
    }

    train_model(cnn, train_loader, cv_loader, n_epochs=100, max_batches=np.inf, model_params=model_params, 
        preprocessing_params=preprocessing_params, model_path=model_path, visualize=True)

    ### evalute
    evaluate(cnn, model_path, test_loaders, visualize=True)
    # name = "cross_2020-05-30_02:31"
    # cnn = torch.load(f"models/grid_search/"+name+"/CNN.pt")
    # model_df = pd.DataFrame()
    # model_df["Learning Rate"] = [1e-3]
    # model_df["Cross-Validation Loss"] = [best_cv_loss]
    # model_df["Model Type"] = [model_params["model_name"]]
    # model_df.to_latex(open(f"{model_path}/model_config.txt", "w"), index=False)

    ### grid search
    # TODO save model params and such in latex table as well!
    model_df = pd.DataFrame()
    lr_grid_search(model_df, test_loaders)
    # TODO set up grid searches and let them run over night!

