import os
from copy import deepcopy

import matplotlib
import numpy as np
import pandas as pd


class Summarization(object):
    """Some summarization examples.

    Input shape (n, m) where n is the number of events in a batch and
    m the number of considered variables.

    The returned (numpy) array must be able to be manipulated by a
    1D numpy boolean mask with the shape m
    """

    @staticmethod
    def abs_mean(data):
        return data.abs().mean(axis=0).cpu().detach().numpy()

    @staticmethod
    def mean(data):
        return data.mean(axis=0).cpu().detach().numpy()

    @staticmethod
    def passtrough(data):
        return data.T.cpu().detach().numpy()

    @staticmethod
    def mean_std(data):
        mean = data.mean(axis=0).cpu().detach().numpy()
        std = data.std(axis=0).cpu().detach().numpy()
        return np.array(list(zip(mean, std)))

    @staticmethod
    def quantile(data):
        lower_quantile, upper_quantile = 0.05, 0.95
        data = data.cpu().detach().numpy()
        return np.array(
            list(
                zip(
                    np.quantile(data, lower_quantile, axis=0),
                    np.quantile(data, 0.5, axis=0),
                    np.quantile(data, upper_quantile, axis=0),
                )
            )
        )


# helper function for saving items
def save_item(item, path, prefix=None, postfix=None):
    if not isinstance(path, list):
        if prefix or postfix:
            directory, filename = os.path.split(path)
            if prefix:
                filename = f"{prefix}_{filename}"
            if postfix:
                basename, extension = os.path.splitext(filename)
                filename = f"{basename}_{postfix}{extension}"
            path = os.path.join(directory, filename)
        if isinstance(item, matplotlib.figure.Figure):
            item.savefig(path, bbox_inches="tight")
        elif isinstance(item, dict):
            np.savez(path, **item)
        elif isinstance(item, pd.DataFrame) and os.path.splitext(path)[1] == ".csv":
            item.to_csv(path)
        elif isinstance(item, pd.DataFrame) and os.path.splitext(path)[1] == ".feather":
            item.columns = [f"{it}" for it in item.columns]
            item.reset_index(inplace=True)
            item.to_feather(path)
    if isinstance(path, list):
        for p in path:
            save_item(item=item, path=p, prefix=prefix, postfix=postfix)


def save_stacked_single_value_tc_point(tc_object, variable_names, path):
    # TODO: generalize it
    for node, _dataframe in tc_object._tc_point.items():
        _dataframe = _dataframe.astype(float)
        _stacked_dataframe = pd.DataFrame()

        _stacked_dataframe["TC Index"] = _dataframe.columns
        _stacked_dataframe["TC Variables"] = [tuple(np.array(variable_names)[np.array(idx)]) for idx in _dataframe.columns]
        _stacked_dataframe["TC Value"] = _dataframe.values[0]

        if sorted:
            _stacked_dataframe.sort_values(by="TC Value", ascending=False, inplace=True, key=abs)

        prefix = f'node_{"_".join(map(str, node)) if isinstance(node, tuple) else node}'

        _csv_path = path if isinstance(path, str) else path[0]
        _csv_path = f"{os.path.splitext(_csv_path)[0]}.csv"

        save_item(_stacked_dataframe, _csv_path, prefix=prefix)


def get_external_representation(tc_collection_dict, convert_to_single_point=False, summarization_function=None):

    new_representation = {}

    for key, df in tc_collection_dict.items():
        _df = deepcopy(df.stack())
        _df = _df.apply(pd.Series).stack().reorder_levels([1, 2, 0]).unstack().T

        if summarization_function:  # renaming if predifined functions are used
            if summarization_function == Summarization.quantile:
                _df.rename(columns={0: "lower quantile", 1: "median", 2: "upper quantile"}, level=1, inplace=True)
            elif summarization_function == Summarization.mean_std:
                _df.rename(columns={0: "mean", 1: "std"}, level=1, inplace=True)
            elif summarization_function == Summarization.mean or summarization_function == Summarization.abs_mean:
                _df.rename(columns={0: "mean"}, level=1, inplace=True)

        if convert_to_single_point:
            _df = _df.T.stack().droplevel(2).unstack().T

        new_representation[key] = _df
    return new_representation
