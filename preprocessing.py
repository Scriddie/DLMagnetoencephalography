# TODO z-normalize every time step

# TODO downsampling
# Select only a few / merge a bunch?
# does it make sense to train only on the differences between
# time steps?
# auto-encoder architecture: big->small->big->small
# select which differences to keep, then use those?

# TODO Batch data

# TODO training time memory management

# IDEAS:
# Signal Space Projection
# Independent Component Analysis

import h5py
import numpy as np
import scipy.special as spec
import os
import torch
import itertools
from sklearn.preprocessing import StandardScaler, scale
import matplotlib.pyplot as plt
np.random.seed(0)
torch.manual_seed(0)

def get_data_paths(type="Intra", train_test="train"):
    assert type in ["Intra", "Cross"], "Not sure if Intra or Cross"
    dataset_names = os.listdir(f"Data/{type}/{train_test}")
    path_names = [os.path.join("Data", type, train_test, i) for i in dataset_names]
    return path_names

def load_data(path):
    # TODO allow loading of multiple data sets
    filename_without_dir = path.split('/')[-1]
    temp = filename_without_dir.split('_')[:-1]
    dataset_name = "_".join(temp)
    with h5py.File(path,'r') as f:
        matrix = f.get(dataset_name)[()]
    return matrix


def create_windows(data, window_size, keep_fraction):
    """ 
    slice data into slices of window_size 
    @param keep_fraction: fraction of windows to keep
    """
    windows = []
    n_possible = data.shape[1] - window_size +1
    for i in range(n_possible):
        if np.random.uniform() < keep_fraction:
            windows.append(data[:, i:i+window_size])
    return windows

def create_labels(train_data, path):
    """ 
    Extract labels from path, arange in form of train_data
    """
    label_dict = {
        "rest": 0,
        "task_motor": 1,
        "task_story": 2,
        "task_story_math": 3,
        "task_working_memory": 4
    }
    label = label_dict["_".join(path.split("/")[-1].split("_")[:-2])]
    labels = [label] * len(train_data)
    return labels

def scale_obs(obs):
    return [scale(i, axis=1) for i in obs]

def preprocess(exp_type="Intra", train_test="train", keep_fraction=0.0002, scale_observations=True):
    # create x of shape (n_batch, n_channel, n_obs)
    dataset_names = get_data_paths(exp_type, train_test)
    datasets = [load_data(i) for i in dataset_names]

    all_windows = []
    all_labels = []
    for i in range(len(datasets[:])):
        data_path = dataset_names[i]
        data = datasets[i]
        window = create_windows(data, 1000, keep_fraction=keep_fraction)
        all_windows.append(window)
        labels = create_labels(window, data_path)
        all_labels.append(labels)
    
    # flatten nested lists
    window_list = list(itertools.chain.from_iterable(all_windows))
    label_list = list(itertools.chain.from_iterable(all_labels))
    
    # scale each channel in each window
    if scale_observations:
        window_list = scale_obs(window_list)

    # shuffle obs and labels equally
    indices = list(range(len(label_list)))
    np.random.shuffle(indices)
    window_list = [window_list[i] for i in indices]
    label_list = [label_list[i] for i in indices]

    # convert to torch tensors
    x_tens = torch.FloatTensor(window_list)
    y_tens = torch.IntTensor(label_list)

    return x_tens, y_tens

def plot_some_obs(x_tens):
    data = x_tens[0]
    print(data[0:3, 0:10])

    for i in range(10):
        plt.plot(range(data.shape[1]), data[i, :])
    plt.show()


if __name__ == "__main__":
    dataset_names = get_data_paths(type="Intra", train_test="train")
    datasets = [load_data(i) for i in dataset_names]
    windows = create_windows(data, 1000)