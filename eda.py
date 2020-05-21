import preprocessing

data = preprocessing.load_data("Data/Intra/train/rest_105923_1.h5")

import matplotlib.pyplot as plt

for i in range(10):
    plt.plot(range(data.shape[1]), data[i, :])