import os

import matplotlib
import numpy as np
import pandas as pd


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
        elif isinstance(item, pd.DataFrame) and os.path.splitext(path) == "csv":
            item.to_csv(path)
        elif isinstance(item, pd.DataFrame) and os.path.splitext(path) in ["pkl", "pickel"]:
            item.to_pickle(path)
    if isinstance(path, list):
        for p in path:
            save_item(item=item, path=p, prefix=prefix, postfix=postfix)


def save_stacked_single_value_tc_point(tc_object, variable_names, path):
    for node, _dataframe in tc_object._tc_point.items():
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
