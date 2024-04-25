import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from torch import nn
from itertools import product


def generate_combinations(numbers, length):
    return list(product(numbers, repeat=length))


mean_sig = [0.5, 0.5]
cov_sig = [[1.0, 0.0], [0.0, 1.0]]

mean_sig2 = [1, -1]
cov_sig2 = [[0.5, 0.0], [0.0, 1.0]]

mean_bkg = [-1.5, -1.5]
cov_bkg = [[1.0, 0.0], [0.0, 1.0]]


def gen_data(
    n=5000,
    mode="multiclass",
    mean_sig=mean_sig,
    cov_sig=cov_sig,
    mean_bkg=mean_bkg,
    cov_bkg=cov_bkg,
):

    X_sig = np.random.multivariate_normal(mean_sig, cov_sig, n)
    X_bkg = np.random.multivariate_normal(mean_bkg, cov_bkg, n)
    y_sig = np.ones(n)
    y_bkg = np.zeros(n)
    if mode == "multiclass":
        x_sig2 = np.random.multivariate_normal(mean_sig2, cov_sig2, n)
        y_sig2 = np.ones(n) * 2

        x = np.concatenate([X_sig, X_bkg, x_sig2])
        y = np.concatenate([y_sig, y_bkg, y_sig2])
    elif mode == "binary":
        x = np.concatenate([X_sig, X_bkg])
        y = np.concatenate([y_sig, y_bkg])
    else:
        raise ValueError("Mode must be either 'multiclass' or 'binary'")
    return x, y


def plot_data(x, y):
    # if three classes in y
    alpha = 0.5
    fill = True
    levels = 5
    if len(np.unique(y)) == 3:
        sns.kdeplot(
            x=x[y == 0, 0],
            y=x[y == 0, 1],
            cmap="Reds",
            alpha=alpha,
            fill=fill,
            levels=levels,
        )
        sns.kdeplot(
            x=x[y == 1, 0],
            y=x[y == 1, 1],
            cmap="Blues",
            alpha=alpha,
            fill=fill,
            levels=levels,
        )
        sns.kdeplot(
            x=x[y == 2, 0],
            y=x[y == 2, 1],
            cmap="Greens",
            alpha=alpha,
            fill=fill,
            levels=levels,
        )
    else:
        sns.kdeplot(
            x=x[y == 0, 0],
            y=x[y == 0, 1],
            cmap="Reds",
            alpha=alpha,
            fill=fill,
            levels=levels,
        )
        sns.kdeplot(
            x=x[y == 1, 0],
            y=x[y == 1, 1],
            cmap="Blues",
            alpha=alpha,
            fill=fill,
            levels=levels,
        )
    plt.grid()
    plt.title("Data")
    # plt.savefig("data.png", bbox_inches="tight")
    # plt.clf()


class Mlp(nn.Module):
    def __init__(self, input_neurons, hidden_neurons, output_neurons, hiddenlayers):

        nn.Module.__init__(self)

        # mlp layers
        self.mlplayers = nn.ModuleList([nn.Linear(input_neurons, hidden_neurons)])
        self.mlplayers.extend(
            [nn.Linear(hidden_neurons, hidden_neurons) for i in range(hiddenlayers + 1)]
        )
        self.mlplayers.append(nn.Linear(hidden_neurons, output_neurons))

    def forward(self, x):
        # input shape: (batch, features)
        for mlplayer in self.mlplayers[:-1]:
            x = mlplayer(x)
            x = torch.tanh(x)

        # new x: (batch, 1)
        x = self.mlplayers[-1](x)
        x = x.squeeze(-1)  # new x: (batch)
        x = torch.sigmoid(x)
        return x


def get_feature_combis(feature_list, combination_list):
    feature_combinations = []
    for combination in combination_list:
        feature_combi = tuple(feature_list[val] for val in combination)
        feature_combinations.append(feature_combi)
    return feature_combinations
