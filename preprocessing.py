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


def load_data(path):
    # TODO allow loading of multiple data sets
    filename_without_dir = path.split('/')[-1]
    temp = filename_without_dir.split('_')[:-1]
    dataset_name = "_".join(temp)
    with h5py.File(path,'r') as f:
        matrix = f.get(dataset_name)[()]
    return matrix


def create_windows(data, window_size):
    """ slice data into slices of window_size """
    windows = []
    # TODO how many windows can we form=?
    # for i in (data.shape[1] - )
    pass

def create_labels(train_data, path):
    """ 
    Extract labels from path, arange in form of train_data
    """
    label_dict = {
        "rest": 0,
        "task_motor": 1,
        "task_story": 2,
        "task_working_memory": 3
    }
    label = label_dict["".join(path.split("/")[-1].split("_")[:-2])]
    labels = np.tile(label, train_data.shape[1])
    return labels