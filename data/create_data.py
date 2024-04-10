#!/usr/bin/env python

######################################################
# This Script is taken from Stefan Wunsch
######################################################

import argparse
import gzip
import os
import pickle
from copy import deepcopy

import numpy as np
#import wget

np.random.seed(1234)

import matplotlib as mpl

mpl.use('Agg')
#mpl.rcParams['font.size'] = 20
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import preprocessing

url_atlas_higgs_challenge = 'http://opendata.cern.ch/record/328/files/atlas-higgs-challenge-2014-v2.csv.gz'


def multivariate_normal(mean_sig, cov_sig, mean_bkg, cov_bkg, num_train,
                        num_test):
    # Signal
    train_sig = np.random.multivariate_normal(mean_sig, cov_sig, num_train)
    test_sig = np.random.multivariate_normal(mean_sig, cov_sig, num_test)

    # Background
    train_bkg = np.random.multivariate_normal(mean_bkg, cov_bkg, num_train)
    test_bkg = np.random.multivariate_normal(mean_bkg, cov_bkg, num_test)

    # Plot processes
    if len(mean_sig) == 2:
        for process, label in zip([train_sig, train_bkg],
                                  ["signal", "background"]):
            plt.figure(figsize=(7, 7))
            plt.hist2d(
                process[:, 0],
                process[:, 1],
                bins=50,
                range=((-4, 4), (-4, 4)),
                cmap=cm.coolwarm)
            plt.xlabel('$x_1$')
            plt.ylabel('$x_2$')
            plt.savefig("{}.png".format(label))
            plt.savefig("{}.pdf".format(label))

        plt.figure(figsize=(2, 2))
        for process, label, cmap in zip([train_bkg,
                                         train_sig], ["background", "signal"],
                                        [plt.cm.Blues, plt.cm.Reds]):
            counts, ybins, xbins, image = plt.hist2d(
                process[:, 0],
                process[:, 1],
                bins=20,
                range=((-3, 3), (-3, 3)),
                cmap=cmap,
                alpha=0.0)
            plt.contour(
                counts,
                extent=[xbins.min(),
                        xbins.max(),
                        ybins.min(),
                        ybins.max()],
                linewidths=3,
                cmap=cmap,
                alpha=0.7)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.xticks([-2, -1, 0, 1, 2])
        plt.yticks([-2, -1, 0, 1, 2])
        plt.savefig("data.png".format(label), bbox_inches="tight")
        plt.savefig("data.pdf".format(label), bbox_inches="tight")

    # Stack datasets
    x_train = np.vstack([train_sig, train_bkg])
    y_train = np.hstack([np.ones(num_train), np.zeros(num_train)])
    w_train = np.ones(2 * num_train)

    x_test = np.vstack([test_sig, test_bkg])
    y_test = np.hstack([np.ones(num_test), np.zeros(num_test)])
    w_test = np.ones(2 * num_test)

    # Perform preprocessing for NN training
    preproc = preprocessing.StandardScaler()
    preproc.fit(x_train)

    x_train = preproc.transform(x_train)

    x_test_nopreproc = deepcopy(x_test)
    x_test = preproc.transform(x_test)

    # Write header
    header = ["x", "y", "z"][0:len(mean_sig)]

    return {
        "header": header,
        "preproc": preproc,
        "x_train": x_train,
        "y_train": y_train,
        "w_train": w_train,
        "x_test": x_test,
        "x_test_nopreproc": x_test_nopreproc,
        "y_test": y_test,
        "w_test": w_test
    }


def load_higgs():
    filename = os.path.basename(url_atlas_higgs_challenge)
    #if not os.path.exists(filename):
    #    wget.download(url_atlas_higgs_challenge)

    # Read data
    with gzip.open(filename) as f:
        lines = f.readlines()

    # Get header
    header = lines[0].decode("utf-8").strip().split(',')[
        1:-4]  # skip everything that is not an input variable
    print('Header:')
    for name in header:
        print(name)

    # Count number of entries
    num_train = 0
    num_test = 0
    for line in lines[1:]:
        set_ = line.decode("utf-8").strip().split(',')[-2]
        if set_ in ['t']:
            num_train += 1
        elif set_ in ['b', 'v']:
            num_test += 1
    print(
        'Number of training/testing events: {}/{}'.format(num_train, num_test))

    # Read out data
    x_train = np.zeros((num_train, len(header)))
    y_train = np.zeros((num_train, 1))
    w_train = np.zeros(num_train)
    x_test = np.zeros((num_test, len(header)))
    y_test = np.zeros((num_test, 1))
    w_test = np.zeros(num_test)
    i_train = 0
    i_test = 0
    for line in lines[1:]:
        set_ = line.decode("utf-8").strip().split(',')[-2]
        type_ = line.decode("utf-8").strip().split(',')[-3]
        weigth = line.decode("utf-8").strip().split(',')[-4]
        values = line.decode("utf-8").strip().split(',')[1:-4]
        if set_ in ['t']:
            for i_v, v in enumerate(values):
                x_train[i_train, i_v] = float(v)
                w_train[i_train] = float(weigth)
                if 's' in type_:
                    y_train[i_train] = 1
                else:
                    y_train[i_train] = 0
            i_train += 1

        elif set_ in ['b', 'v']:
            for i_v, v in enumerate(values):
                x_test[i_test, i_v] = float(v)
                w_test[i_test] = float(weigth)
                if 's' in type_:
                    y_test[i_test] = 1
                else:
                    y_test[i_test] = 0
            i_test += 1

    # Replace missing values
    np.place(x_train, x_train == -999.0, -10.0)
    np.place(x_test, x_test == -999.0, -10.0)

    # Perform preprocessing for NN training
    preproc = preprocessing.StandardScaler()
    preproc.fit(x_train)

    x_train = preproc.transform(x_train)

    x_test_nopreproc = deepcopy(x_test)
    x_test = preproc.transform(x_test)

    # Inject "wrong" data
    i1 = header.index('DER_mass_vis')
    i2 = header.index('DER_mass_MMC')
    print('Indices:', i1, i2)
    mask_train = y_train.squeeze()
    mask_test = y_test.squeeze()
    r = np.random.randn(600000)
    r2 = np.random.randn(600000)

    for i in range(x_train.shape[0]):
        x_train[i, i1] = r[i]
        if mask_train[i] == 1:
            x_train[i, i2] = r2[i]
        else:
            x_train[i, i2] = r2[i]

    for i in range(x_test.shape[0]):
        x_test[i, i1] = r[i]
        if mask_test[i] == 1:
            x_test[i, i2] = r2[i]
        else:
            x_test[i, i2] = r2[i]

    return {
        "header": header,
        "preproc": preproc,
        "x_train": x_train,
        "y_train": y_train,
        "w_train": w_train,
        "x_test": x_test,
        "x_test_nopreproc": x_test_nopreproc,
        "y_test": y_test,
        "w_test": w_test
    }


def parse_arguments():
    parser = argparse.ArgumentParser(description="Create datasets.")
    parser.add_argument("--scenario", required=True, type=str, help="Scenario")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Create dataset
    scenario = args.scenario

    # yapf: disable
    if scenario == "A":
        mean_sig = [0.5, 0.5]
        cov_sig = [[1.0, 0.0],
                   [0.0, 1.0]]

        mean_bkg = [-0.5, -0.5]
        cov_bkg = [[1.0, 0.0],
                   [0.0, 1.0]]

    elif scenario == "A2":
        mean_sig = [0.5, 0.25]
        cov_sig = [[1.0, 0.0],
                   [0.0, 1.0]]

        mean_bkg = [-0.5, -0.25]
        cov_bkg = [[1.0, 0.0],
                   [0.0, 1.0]]

    elif scenario == "B":
        mean_sig = [0.0, 0.0]
        cov_sig = [[1.0, 0.5],
                   [0.5, 1.0]]

        mean_bkg = [0.0, 0.0]
        cov_bkg = [[1.0, -0.5],
                   [-0.5, 1.0]]

    elif scenario == "C":
        mean_sig = [0.5, 0.5]
        cov_sig = [[1.0, 0.5],
                   [0.5, 1.0]]

        mean_bkg = [-0.5, -0.5]
        cov_bkg = [[1.0, -0.5],
                   [-0.5, 1.0]]

    elif scenario == "D":
        mean_sig = [0.0, 0.0]
        cov_sig = [[0.5, 0.0],
                   [0.0, 0.5]]

        mean_bkg = [0.0, 0.0]
        cov_bkg = [[3.0, 0.0],
                   [0.0, 3.0]]

    elif scenario == "E":
        mean_sig = [0.5, 0.5, 0.0]
        cov_sig = [[1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.7],
                   [0.0, 0.7, 1.0]]

        mean_bkg = [-0.5, -0.5, 0.0]
        cov_bkg = [[1.0, 0.0, 0.0],
                   [0.0, 1.0, -0.7],
                   [0.0, -0.7, 1.0]]

    # yapf: enable

    if scenario != "HIGGS":
        obj = multivariate_normal(
            mean_sig,
            cov_sig,
            mean_bkg,
            cov_bkg,
            num_train=100000,
            num_test=100000)
            #num_train=10000000,
            #num_test=10)
    else:
        obj = load_higgs()
    # Write to file
    pickle.dump(obj, open("data.pickle", "wb"))
