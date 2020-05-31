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
import shutil
import matplotlib.ticker as ticker
np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cpu")



def train_model(model, train_loader, cv_loader, n_epochs, max_batches, model_params,
    preprocessing_params, model_path, visualize, close_plots=None):
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
        n_cv = 0
        cv_acc = 0.
        # batch gradient descent
        for n_batch, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            model_params["optimizer"].zero_grad()
            output = model.forward(x_batch)
            train_acc += multi_acc(output, y_batch) * x_batch.size(0)
            loss = model_params["loss_fn"](output, y_batch.long())
            total_train_loss += loss.item()
            loss.backward()
            model_params["optimizer"].step()

            with torch.no_grad():
                x_batch_cv, y_batch_cv = next(iter(cv_loader))
                x_batch_cv = x_batch_cv.to(device)
                y_batch_cv = y_batch_cv.to(device)
                cv_preds = model(x_batch_cv, dropout=False)
                cv_loss = model_params["loss_fn"](cv_preds, y_batch_cv.long())
                total_cv_loss += cv_loss.item()
                n_cv += x_batch_cv.size(0)
                cv_acc += multi_acc(cv_preds, y_batch_cv) * x_batch_cv.size(0)
                
        train_losses.append(total_train_loss)
        cv_losses.append(total_cv_loss)
        epoch_train_acc = train_acc / len(train_loader.dataset)
        epoch_cv_acc = cv_acc / n_cv
        if ((epoch_train_acc == 1) or (round(total_train_loss, 3) == 0)) and (close_plots is None):
            print("Training Accuracy 100%")
            break
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

    if not (close_plots is None):
        return pd.DataFrame({
            "epochs": list(range(epoch+1)),
            "train_losses": train_losses,
            "cv_losses": cv_losses
        })

    return best_cv_acc.item()


def lr_grid_search(model_df, test_loaders):
    # TODO grid search learning rate
    learning_rates = np.linspace(1e-5, 1e-1, 8)
    model_path = f"models/grid_search/{type_name}_" + datetime.now().strftime("%Y-%m-%d_%H:%M")[0:16]
    os.mkdir(model_path)
    all_accuracies = []
    for lr in learning_rates:
        print(f"Learning Rate: {lr}")
        model = models.simpleCNN(input_dim=248, output_dim=4)
        model.to(device)
        model_params = {
            "n_samples": x_tens.size(0),
            "loss_fn": torch.nn.CrossEntropyLoss(),
            "optimizer": torch.optim.Adam(model.parameters(), lr=lr),
            "early_stopping": 5,
            "model_name": "CNNDropout"
        }
        best_cv_loss = train_model(model, train_loader, cv_loader, n_epochs=200, max_batches=9999, model_params=model_params, 
            preprocessing_params=preprocessing_params, model_path=model_path, visualize=False)
        all_accuracies.append(best_cv_loss)
        # TODO create a dataframe, log best cv_loss
        # TODO get model train & cv acc
    df = pd.DataFrame({
        "Learning Rate": learning_rates,
        "labels": ["{:.2e}".format(i) for i in learning_rates],
        "Validation Accuracy": all_accuracies
    })
    df.to_latex(open(f"{model_path}/lr_grid_search.txt", "w"), index=False)
    plt.close("all")
    # tick = ticker.ScalarFormatter(useOffset=False, useMathText=True)
    # tick.set_powerlimits((0,0))
    g = sns.barplot(data=df, x="Learning Rate", y="Validation Accuracy")
    g.set_xticklabels(df.labels, rotation=30)
    plt.title("Learning Rate Tuning")
    plt.tight_layout()
    plt.savefig(f"{model_path}/lr_grid_search.png", dpi=500)
        # model_df["Learning Rate"] = [lr]
        # model_df["Cross-Validation Accuracy"] = [best_cv_loss]
        # model_df["Model Type"] = [model_params["model_name"]]
        # model_df.to_latex(open(f"{model_path}/model_config.txt", "w"), index=False)
        # evaluate(model, model_path, test_loaders)

def initializations():
    model_path = f"models/grid_search/{type_name}_initial_" + datetime.now().strftime("%Y-%m-%d_%H:%M")[0:16]
    os.mkdir(model_path)
    # all_accuracies = []
    reload(models)
    plot_dfs = []
    schemes = ["xavier_normal", "random_uniform", "ones"]
    for idx, i in enumerate(schemes):
        print(f"Initialization: {i}")
        model = models.simpleCNNInitial(input_dim=248, output_dim=4, init=i)
        model.to(device)
        model_params = {
            "n_samples": x_tens.size(0),
            "loss_fn": torch.nn.CrossEntropyLoss(),
            "optimizer": torch.optim.Adam(model.parameters(), lr=1e-3),
            "early_stopping": 50,
            "model_name": "CNNDropout"
        }
        plot_df = train_model(model, train_loader, cv_loader, n_epochs=30, max_batches=9999, model_params=model_params, 
            preprocessing_params=preprocessing_params, model_path=model_path, visualize=False, close_plots=True)
        plot_dfs.append(plot_df)
    
    plt.close("all")
    for i in plot_dfs:
        plt.plot(i["epochs"], i["train_losses"])
    plt.legend(schemes)
    plt.title("Initialization Schemes")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.grid()
    plt.savefig(model_path + "/init_schemes.png", dpi=500)
    plt.show()

    # TODO xavier normal
    # random uniform
    pass



def evaluate(model, model_path, test_loaders, visualize=False):
    all_accs = pd.DataFrame()
    for i in range(1, 1+len(test_dirs)):
        acc_df = evaluate_sample(model, test_loaders[i-1], test_label_dict, model_path=model_path,
             title=f"{type_name} Subject - test{i}", intra_cross=type_name, visualize=visualize)
        acc_df["subject"] = [f"test{i}"]
        all_accs = pd.concat((all_accs, acc_df), axis=0)
    all_accs.to_latex(open(f"{model_path}/accuracy_table.txt", "w"), index=False)
    # TODO create a lil bar plot for all test dirs

def train_intra():
    pass


if __name__ == "__main__":

    ### Intra
    # type_name = "Intra"
    # load_intra_cv = False

    ### Cross
    type_name = "Cross"
    mode = "init" # ["train", "search", "init"]
    
    if type_name == "Intra":
        load_intra_cv = False
        cnn = models.simpleCNN  # 100% on Intra
        test_dirs = ["test"]
        train_fraction = 0.8
        n_epochs = 10
    else:
        load_intra_cv = True
        test_dirs = ["test1", "test2", "test3"]
        train_fraction = 1.0
        n_epochs = 100

    ### load data
    preprocessing_params = {
        "downsampling": 10, 
        "window_size": 1000, 
        "keep_fraction": 0.01, 
        "scale_observations": True}
    all_tasks = pp.load_data(type_name=type_name, train_test="train", downsampling=preprocessing_params["downsampling"])
    x_tens, y_tens, label_dict = pp.preprocess(all_tasks, **preprocessing_params)

    preparation_params = {
        "train_fraction": train_fraction,
        "batch_size": 16,
        "n_classes": 4,
        "model_type": "cnn"}
    train_loader, cv_loader = prepare_data(x_tens, y_tens, **preparation_params)
    if load_intra_cv:
        cv_tasks = pp.load_data(type_name=type_name, train_test="CV", downsampling=preprocessing_params["downsampling"])
        cv_x_tens, cv_y_tens, cv_label_dict = pp.preprocess(cv_tasks, **preprocessing_params)
        cv_preparation_params = {
            "train_fraction": 0,
            "batch_size": 16,
            "n_classes": 4,
            "model_type": "cnn"}
        _, cv_loader = prepare_data(cv_x_tens, cv_y_tens, **cv_preparation_params)

    test_loaders = []
    for test_dir in test_dirs:
        test_subjects = pp.load_data(type_name=type_name, train_test=test_dir, downsampling=preprocessing_params["downsampling"])
        x_test, y_test, test_label_dict = pp.preprocess(test_subjects, **preprocessing_params)
        tl, _ = prepare_data(x_test, y_test, **preparation_params)
        test_loaders.append(tl)

    ### train cnn
    if mode == "train":
        # TODO iterate over all models once they're done
        # cnn = models.CNN(input_dim=248, output_dim=4)
        reload(models)
        # all_models = [
        #     models.simpleCNN(input_dim=248, output_dim=4),
        #     models.deepCNN(input_dim=248, output_dim=4),
        #     models.deepCNNDropout(input_dim=248, output_dim=4),
        #     models.deepCNNBatchNorm(248, 4),
        #     models.CNN2D(248, 4),
        #     models.CNNSideways(248, 4),
        # ]
        # for cnn in all_models:
        reload(models)
        # cnn = models.CNN2D(248, 4)
        cnn = models.deepCNNDropout(input_dim=248, output_dim=4)
        # cnn = models.simpleCNN(input_dim=248, output_dim=4)
        cnn.to(device)
        
        lr = 1e-6
        model_params = {
            "n_samples": x_tens.size(0),
            "loss_fn": torch.nn.CrossEntropyLoss(),
            "optimizer": torch.optim.Adam(cnn.parameters(), lr=lr),
            "early_stopping": 5,
            "model_name": cnn.model_name
        }

        model_path = f"models/{type_name}_"+model_params["model_name"]
        #  + datetime.now().strftime("%Y-%m-%d_%H:%M")[0:16]
        try:
            os.mkdir(model_path)
        except FileExistsError:
            pass

        best_cv_acc = train_model(cnn, train_loader, cv_loader, n_epochs=n_epochs, max_batches=np.inf, model_params=model_params, 
            preprocessing_params=preprocessing_params, model_path=model_path, visualize=True)

        ### evalute
        evaluate(cnn, model_path, test_loaders, visualize=True)
        model_df = pd.DataFrame()
        model_df["Learning Rate"] = [1e-3]
        model_df["Cross-Validation Accuracy"] = [best_cv_acc]
        model_df["Model Name"] = [model_params["model_name"]]
        model_df.to_latex(open(f"{model_path}/model_config.txt", "w"), index=False)

    # ### grid search
    if mode == "search":
        model_df = pd.DataFrame()
        lr_grid_search(model_df, test_loaders)

    if mode == "init":
        initializations()

