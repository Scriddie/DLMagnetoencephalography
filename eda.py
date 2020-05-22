import preprocessing as pp
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import itertools

np.random.seed(0)
torch.manual_seed(0)


class ANN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # TODO think about how to scale up architecture
        # self.hidden_dims = [10]
        self.fc1 = nn.Linear(input_dim, 10)
        self.fc2 = nn.Linear(10, output_dim)
        # self.softmax = torch.nn.Softmax(10, output_dim)

    def forward(self, x):
        output = self.fc1(x)
        output = self.fc2(output)
        # output = self.softmax(output)
        return(output)

# TODO create some dummy data
x = np.random.random((10, 3))
# x = np.array([1,2,3]).reshape(1, -1)
# y = x
y = x[:, 1].reshape(-1, 1)

x = torch.FloatTensor(x)
y = torch.FloatTensor(y)

# TODO learn identity function
ann = ANN(3, 1)

loss_fn = torch.nn.MSELoss(reduction="mean")
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


# TODO get the actual data
x_tens, y_tens = pp.preprocess()

# TODO define some simple as network and do categorical class predictions
# ohe class labels, use BCEloss

# find number of classes in data set:
n_classes = len(np.unique(y_tens))
ex_label = y_tens[0]
# ex_label is tensor(3.)

# TODO make sure our labels are a tensor of ints!

labels = torch.tensor([3., 0., 1., 3., 0., 4.])

a = y_tens[0:6].numpy()
a = torch.tensor(a)
a = a.unsqueeze(0)
target = torch.zeros(a.size(0), 15).scatter_(1, a, 1.)
ex_label = y_tens[0]
# ex_label is tensor(3.)
# TODO report pytorch
target = torch.zeros(1, n_classes).scatter_(0, ex_label, 1.)

labels = torch.tensor([3, 0, 1, 3, 0, 4])
labels = labels.unsqueeze(0)
target = torch.zeros(labels.size(0), 15).scatter_(1, labels, 1.)