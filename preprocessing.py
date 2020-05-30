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
from scipy.signal import decimate
import glob
np.random.seed(0)
torch.manual_seed(0)

# def get_data_paths(type_name="Intra", train_test="train"):
#     assert type in ["Intra", "Cross"], "Not sure if Intra or Cross"
#     dataset_names = os.listdir(f"Data/{type_name}/{train_test}")
#     path_names = [os.path.join("Data", type_name, train_test, i) for i in dataset_names]
#     return path_names

# def load_data(path):
#     # TODO allow loading of multiple data sets
#     filename_without_dir = path.split('/')[-1]
#     temp = filename_without_dir.split('_')[:-1]
#     dataset_name = "_".join(temp)
#     with h5py.File(path,'r') as f:
#         matrix = f.get(dataset_name)[()]
#     return matrix

def load_data(type_name, train_test, downsampling):
    """ yield training data on a per-task level """
    directory = f"Data/{type_name}/{train_test}/"
    dataset_names = os.listdir(f"Data/{type_name}/{train_test}")

    # identify all subjects and tasks
    all_subjects = []
    subjects = np.unique([i.split("_")[-2] for i in dataset_names])
    tasks = np.unique(["_".join(i.split("/")[-1].split("_")[:-2])for i in dataset_names])

    for subject in subjects:
        # sorder all subjects and tasks alphabetically to maintain the order
        subject_files = sorted([i for i in dataset_names if subject in i])
        for task in tasks:
            current_task_matrix = None
            task_files = [i for i in subject_files if task in i]
            for file_name in task_files:
                full_name = directory + file_name
                with h5py.File(full_name, "r") as f:
                    matrix = f.get("_".join(file_name.split('_')[:-1]))[()]
                if current_task_matrix is None:
                    current_task_matrix = matrix
                else:
                    current_task_matrix = np.hstack((current_task_matrix, matrix))
            current_task_matrix = decimate(current_task_matrix, q=downsampling)
            yield (subject, task, current_task_matrix)

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

def create_labels(train_data, label_name):
    """ 
    Extract labels from path, arange in form of train_data
    """
    label_dict = {
        "rest": 0,
        "task_motor": 1,
        "task_story_math": 2,
        "task_working_memory": 3
    }
    if len(label_dict) == 0:
        label = 0
        label_dict[label_name] = label
    elif label_name not in label_dict.keys():
        label = max(label_dict.values()) + 1
        label_dict[label_name] = label
    else:
        label = label_dict[label_name]
    labels = [label] * len(train_data)
    return labels, label_dict

def scale_obs(obs):
    return [scale(i, axis=1) for i in obs]

# TODO create cross-validation set alongside
# TODO add in some nice prints about how exactly we are pre-processing data
def preprocess(datasets, window_size, downsampling, keep_fraction, scale_observations):
    """ 
    @param datasets: Generator of datasets to perform windowing on
    @param label_list: a list of labels for each dataset
    @param window_size: number of observations per window
    @param keep_fraction: fraction of windows to keep
    @param scale_observations: whether or not to normalize each window
    """

    all_windows = []
    all_labels = []
    for subject, task, data in datasets:
        print(f"Loading subject {subject} {task}")
        window = create_windows(data, window_size, keep_fraction=keep_fraction)
        all_windows.append(window)
        labels, label_dict = create_labels(window, task)
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

    return x_tens, y_tens, label_dict

def plot_some_obs(x_tens):
    data = x_tens[0]
    print(data[0:3, 0:10])

    for i in range(10):
        plt.plot(range(data.shape[1]), data[i, :])
    plt.show()


if __name__ == "__main__":

    all_subjects = load_data(type_name="Intra", train_test="train")
    datasets = [all_subjects["105923"]["rest"], all_subjects["105923"]["task_motor"]]
    label_list = ["rest", "task_motor"]

    x_tens, y_tens = preprocess(datasets, label_list, window_size=1000, keep_fraction=0.001, scale_observations=True)

