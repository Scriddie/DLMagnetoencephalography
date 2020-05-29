import preprocessing as pp
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import itertools
from scipy.signal import decimate
from sklearn.decomposition import SparsePCA, PCA
import pandas as pd
import seaborn as sns

np.random.seed(0)
torch.manual_seed(0)


class ANN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # TODO think about how to scale up architecture
        # self.hidden_dims = [10]
        self.fc1 = nn.Linear(input_dim, 10)
        self.fc2 = nn.Linear(10, output_dim)
        self.softmax = nn.Softmax()
        # self.softmax = torch.nn.Softmax(10, output_dim)

    def forward(self, x):
        output = self.fc1(x)
        output = self.fc2(output)
        output = self.softmax(output)
        return(output)

def test_ann():
    # TODO create some dummy classification data
    x = np.random.randint(size=(10, 3), low=0, high=5)
    n_classes = 5
    all_cat_labels = np.zeros((x.shape[0], n_classes))
    for i in range(x.shape[0]):
        j = 1
        all_cat_labels[i][j] = 1

    y = torch.Tensor(all_cat_labels)

    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)

    # TODO learn identity function
    ann = ANN(3, 5)

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(ann.parameters(), lr=1e-3)

    losses = []
    n_epochs = 100
    for epoch in range(n_epochs):
        # TODO use optimizer to update the weights
        optimizer.zero_grad()
        output = ann.forward(x)
        loss = loss_fn(output, y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    plt.plot(range(n_epochs), losses)
    plt.show()

def pca():
    # TODO PCA for optimal num channels?
    pass


if __name__ == "__main__":
    all_subjects = pp.load_data(type_name="Intra", train_test="train")
    label_list = ['rest', 'task_motor', 'task_story_math', 'task_working_memory']
    datasets = [all_subjects["105923"]["rest"]]

    preprocessing_params = {
        "downsampling": 50, 
        "window_size": 1000, 
        "keep_fraction": 0.01, 
        "scale_observations": True
    }

    x_tens, y_tens, label_dict = pp.preprocess(datasets, label_list, **preprocessing_params)
    # data = decimate(datasets[0], q=50)
    
    # TODO not sure what this means :/

    # TODO PCA on subsampled x_tens
    data = datasets[0]
    pca = PCA(n_components=10)
    pca.fit(data)
    print(pca.explained_variance_ratio_)

    # TODO plot number of components, cumulative explained variance
    df = pd.DataFrame({
        "component_explained_variance": pca.explained_variance_ratio_,
        "n_components": list(range(len(pca.explained_variance_ratio_)))

    })

    sns.lineplot(data=df, x="n_components", y="component_explained_variance")
    plt.show()

    pass

