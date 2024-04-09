import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Setup Means and covariances of individual classes

mean1 = [1.5, 1.5]
cov1=   [[1.0, 0.5],
        [0.5, 1.0]]

mean2 = [-1.5, -1.5]
cov2 =  [[1.0, 0.5],
        [0.5, 1.0]]

mean3 = [2.5, -2.5]
cov3 =  [[1.0, 0.5],
        [0.5, 1.0]]

colors = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens]

means = [mean1, mean2, mean3]
covs = [cov1, cov2, cov3]

# choose number of events
n = 20000

# create and plot data

# loop over classes
data_list = []

for mean, cov in zip(means, covs):
    data = np.random.multivariate_normal(mean, cov, n)
    data_list.append(data)

for class_n, zipper in enumerate(zip(data_list, colors)):
    data, cmap = zipper[0], zipper[1]

    counts, xedges, yedges = np.histogram2d(data[:, 0],
                                            data[:, 1],
                                            bins=80)
    plt.contour(
        counts,
        extent=[xedges.min(),
                xedges.max(),
                yedges.min(),
                yedges.max()],
        linewidths=2,
        levels=4,
        cmap=cmap,
        alpha=0.7)

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.savefig("multiclass.png")
plt.savefig("multiclass.pdf")

y_data = np.hstack([np.zeros(n), np.ones(n), 2*np.ones(n)])
x_data = np.vstack(data_list)

# Perform preprocessing for NN training
preproc = preprocessing.StandardScaler()
preproc.fit(x_data)
x_data = preproc.transform(x_data)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.4)

pickle.dump({'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}, open("multiclass_data.pickle", "wb"))
