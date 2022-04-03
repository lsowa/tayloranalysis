import os
from itertools import product

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.autograd import grad

matplotlib.rc("font", size=16, family="serif")
lw, markeredgewidth = 3, 3

# helper function for saving items
def save_item(item, _path, prefix=None):
    if not isinstance(_path, list):
        if prefix:
            _dir, _name = os.path.split(_path)
            _path = os.path.join(_dir, f"{prefix}_{_name}")
        if isinstance(item, matplotlib.figure.Figure):
            item.savefig(_path, bbox_inches="tight")
        elif isinstance(item, dict):
            np.savez(_path, **item)
        elif isinstance(item, pd.DataFrame):
            item.to_csv(_path)
    if isinstance(_path, list):
        for p in _path:
            save_item(item=item, _path=p, prefix=prefix)


class TaylorAnalysis(object):
    """
    Class to wrap nn.Module for taylorcoefficient analysis.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self._orders = {1: self._first_order, 2: self._second_order, 3: self._third_order}

    def _mean(self, data):
        """Compute abs and mean of taylorcoefficients.

        Args:
            data (torch.tensor): tensor with taylorcoefficients of shape (batch, features)

        Returns:
            numpy.array: Array means of taylorcoefficients.
        """
        data = torch.abs(data)
        data = torch.mean(data, dim=0)
        return data.cpu().detach().numpy()

    def _first_order(self, x_data):
        """Compute first order taylorcoefficients.

        Args:
            x_data (torch.tensor): X data of shape (batch, features).

        Returns:
            torch.tensor: First order taylorcoefficients (batch, features).
        """
        x_data.requires_grad = True
        self.model.zero_grad()
        x_data.grad = None
        pred = self.model(x_data)
        pred = pred.sum()
        # first order grads
        gradients = grad(pred, x_data)
        return self._mean(gradients[0])

    def _second_order(self, x_data, ind_i):
        """Compute second order taylorcoefficients. The model is first derivated according to the ind_i-th feature and second to all others.

        Args:
            x_data (torch.tensor): X data (batch, features).
            ind_i (int): Feature for the first derivative.

        Returns:
            torch.tensor: Second order derivatives according to ind_i and all other input variables (batch, feature).
        """
        x_data.requires_grad = True
        self.model.zero_grad()
        x_data.grad = None
        pred = self.model(x_data)
        pred = pred.sum()
        # first order gradients
        gradients = grad(pred, x_data, create_graph=True)
        gradients = gradients[0].sum(dim=0)
        # second order gradients
        gradients = grad(gradients[ind_i], x_data)
        gradients = gradients[0]
        # factor for all second order taylor terms
        gradients /= 2.0
        # factor for terms who occure two times in the second order (e.g. d/dx1x2 and d/dx2x1)
        masked_factor = torch.tensor(range(gradients.shape[1]), device=gradients.device)
        masked_factor = (masked_factor != ind_i) + 1
        gradients *= masked_factor
        return self._mean(gradients)

    def _third_order(self, x_data, ind_i, ind_j):
        """Compute third order taylorcoefficients. The model is derivated to the ind_i-th feature,
            the ind_j-th feature and third to all other features.

        Args:
            x_data (torch.tensor): X data (batch, features).
            ind_i (int): Feature for the first derivative.
            ind_j (int): Feature for the second derivative.

        Returns:
            torch.tensor: Third order derivatives according to ind_i, ind_j and all other input features (batch, feature).
        """
        x_data.requires_grad = True
        self.model.zero_grad()
        x_data.grad = None
        pred = self.model(x_data)
        pred = pred.sum()
        # first order gradients
        gradients = grad(pred, x_data, create_graph=True)
        gradients = gradients[0].sum(dim=0)
        # second order gradients
        gradients = grad(gradients[ind_i], x_data, create_graph=True)
        gradients = gradients[0].sum(dim=0)
        # third order gradients
        gradients = grad(gradients[ind_j], x_data)
        gradients = gradients[0]
        # factor for all third order taylor terms
        gradients /= 6.0
        # factor for all terms that occur three times (e.g. d/dx1x2x2 and d/dx2x1x2 and d/dx2x2x1)
        masked_factor = np.array(range(gradients.shape[1]))
        # check for derivatives with same variables
        masked_factor = (
            torch.tensor(masked_factor == ind_j, dtype=int)
            + torch.tensor(masked_factor == ind_i, dtype=int)
            + torch.tensor([ind_j == ind_i] * masked_factor.shape[0], dtype=int)
        )
        masked_factor = (masked_factor == 1) * 2 + 1  # if variable pair is identical ..
        # print(factor_bool)
        gradients *= masked_factor.to(gradients.device)
        return self._mean(gradients)

    def _get_derivatives(self, option, variable_idx, derivation_order):
        """
        Creates all derivative combinations for the dataframe in which the
        checkpoints are stored or a reduced variant for the calculation, since
        n-plets are calculated there simultaneously.

        Args:
            option (str): "calculation" or "dataframe".
            variable_idx tuple[int]: Variables (indices) used in the derivations
            derivation_order (int): Highest order of derivatives

        Returns:
            list[tuple[int]]: List of derivations in the form of tuples.
        """
        _derivatives = [()] if option == "calculation" else [(idx,) for idx in variable_idx]
        _order = 2
        while _order <= derivation_order:
            _j = _order - 1 if option == "calculation" else _order
            _tmp = [list(it) for it in product(*[variable_idx for _ in range(_j)])]
            _tmp = np.unique(np.sort(_tmp, axis=1), axis=0).tolist()
            _derivatives += list(map(tuple, _tmp))
            _order += 1
        return _derivatives

    def _get_empty_checkpoints_dataframe(self, derivatives):
        """
        Creates an empty dataframe for the checkpoints

        Args:
            derivatives list[tuple[int]]: List of derivations in the form of tuples.

        Returns:
            pd.DataFrame: empty pandas.DataFrame.
        """
        _df = pd.DataFrame(data=None, columns=["Epoch"] + derivatives)
        _df.set_index("Epoch", inplace=True)
        _df = _df.astype(float)
        return _df

    def _tc_checkpoint(self, x_data, epoch, dataframe, variable_mask, variable_idx, derivatives_for_calculation):
        """
        Goes through the list of derivatives (reduced variant for the calculation) and writes
        the derivatives into the dataframe.

        Args:
            x_data (torch.tensor): X data (batch, features).
            epoch (int): Current epoch.
            dataframe: (pd.DataFrame): Dataframe with all with all derivation combinations.
            variable_mask (list[int]): Variable indices used in the calculation used for n-plet creation.
            variable_idx (list[int]): Variable indices used in the calculation used to mask out unwanted values from n-plets.
            derivatives_for_calculation (list[tuple[int]]): list of derivatives (reduced variant for the calculation).

        Returns:
            None
        """
        _nplet = lambda *__x: [(idx,) for idx in variable_idx] if __x is None else [(idx, *__x) for idx in variable_idx]
        _mask = lambda __nplet: pd.Series(__nplet).isin(dataframe.columns).to_numpy()

        for item in derivatives_for_calculation:
            nplet = _nplet(*item)
            if any(mask := _mask(nplet)):
                dataframe.loc[epoch, np.array(nplet)[mask]] = self._orders[len(item) + 1](x_data, *item)[variable_mask][mask]

    def tc_checkpoint(self, x_data, epoch):
        """
        Compute and save taylorcoefficients to plot and save them later.

        Args:
            x_data (torch.tensor): X data (batch, features).
            epoch (int): Current epoch.

        Returns:
            None
        """
        self._tc_checkpoint(
            x_data=x_data,
            epoch=epoch,
            dataframe=self._checkpoints,
            variable_mask=self.variable_mask,
            variable_idx=self.variable_idx,
            derivatives_for_calculation=self.derivatives_for_calculation,
        )

    def setup_tc_checkpoints(
        self,
        number_of_variables_in_data,
        considered_variables_idx=None,
        variable_names=None,
        derivation_order=2,
    ):
        """
        Method for setting all important parameters for calculating the checkpoints during training

        Args:
            number_of_variables_in_data (int): Total number of variables present in data.
            considered_variables_idx (list[int]): Contains the indices of variables according to which
                                                  the derivation is required. All variables are
                                                  considered, unless explicitly stated otherwise.
            variable_names (list[str]): Contains the (LaTeX) type names for the plots. If not
                                        otherwise specified defaults are used ["x_1", "x_2", ...].
            derivation_order (int): Highest order of derivatives.
        Returns:
            None
        """

        self.derivation_order = derivation_order
        self.variable_idx = considered_variables_idx or list(range(number_of_variables_in_data))
        self.variable_names = variable_names or [f"x_{i}" for i in range(number_of_variables_in_data)]
        self.variable_mask = np.array(self.variable_idx)

        self.derivatives_for_calculation = self._get_derivatives(
            "calculation",
            variable_idx=self.variable_idx,
            derivation_order=self.derivation_order,
        )
        self.derivatives_for_dataframe = self._get_derivatives(
            "dataframe",
            self.variable_idx,
            derivation_order=self.derivation_order,
        )
        self._checkpoints = self._get_empty_checkpoints_dataframe(self.derivatives_for_dataframe)

    def plot_checkpoints(self, path="./tc_training.pdf"):
        """
        Plot saved checkpoints.

        Args:
            path (str) or (list[str]): /path/to/save/plot.pdf or ["/path/to/save/plot.pdf", "/path/to/save/plot.png"]
        """

        fig_and_ax = [plt.subplots(1, 1, figsize=(10, 7)) for _ in range(self.derivation_order + 1)]
        fig, ax = tuple(zip(*fig_and_ax))  # 0: all, 1: first order, 2: second order...

        for column in self._checkpoints.columns:
            _label = ",".join(np.array(self.variable_names)[np.array(column)])
            _label = f"$<t_{{{_label}}}>$"
            ax[0].plot(self._checkpoints[column], label=_label, lw=lw)
            ax[len(column)].plot(self._checkpoints[column], label=_label, lw=lw)

        for _ax in ax:
            _ax.legend(loc="upper left", bbox_to_anchor=(1.04, 1))
            _ax.set_xlabel("Epoch", loc="right", fontsize=13)
            _ax.set_ylabel("$<t_i>$", loc="top", fontsize=13)
            _ax.yaxis.set_tick_params(which="both", right=True, direction="in")
            _ax.xaxis.set_tick_params(which="both", top=True, direction="in")

        prefix = [""] + [f"order_{i+1}" for i in range(self.derivation_order)]
        for _fig, _pref in zip(fig, prefix):
            save_item(_fig, path, prefix=_pref)

        plt.close("all")

    def plot_taylor_coefficients(
        self,
        x_data,
        considered_variables_idx=None,
        variable_names=None,
        derivation_order=2,
        path="./coefficients.pdf",
    ):
        """
        Plot taylorcoefficients for current weights of the model.

        Args:
            x_data (torch.tensor): X data of shape (batch, features).
            considered_variables_idx (list[int]): Contains the indices of variables according to which
                                                  the derivation is required. All variables are
                                                  considered, unless explicitly stated otherwise.
            variable_names (list[str]): Contains the (LaTeX) type names for the plots. If not
                                        otherwise specified defaults are used ["x_1", "x_2", ...].
            derivation_order (int): Highest order of derivatives.
            path (str) or (list[str]): /path/to/save/plot.pdf or ["/path/to/save/plot.pdf", "/path/to/save/plot.png"]
        """

        variable_idx = considered_variables_idx or list(range(x_data.shape[1]))
        variable_names = variable_names or [f"x_{idx}" for idx in variable_idx]
        variable_mask = np.array(variable_idx)

        _derivatives_calculation = self._get_derivatives(
            "calculation",
            variable_idx=variable_idx,
            derivation_order=derivation_order,
        )
        _derivatives_dataframe = self._get_derivatives(
            "dataframe",
            variable_idx=variable_idx,
            derivation_order=derivation_order,
        )
        _df = self._get_empty_checkpoints_dataframe(_derivatives_dataframe)

        self._tc_checkpoint(
            x_data=x_data,
            epoch=0,
            dataframe=_df,
            variable_mask=variable_mask,
            variable_idx=variable_idx,
            derivatives_for_calculation=_derivatives_calculation,
        )

        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        xlabels = []

        for idx, column in enumerate(_df.columns):
            _label = ",".join(np.array(self.variable_names)[np.array(column)])
            xlabels.append(f"$<t_{{{_label}}}>$")
            ax.plot(idx, _df.loc[0][column], "+", color="black", markersize=10, markeredgewidth=markeredgewidth)

        ax.set_ylabel("$<t_i>$", loc="top")
        ax.set_xticks(list(range(idx + 1)))
        ax.set_xticklabels(xlabels, rotation=45, ha="right", rotation_mode="anchor")
        ax.grid(axis="x", alpha=0.25)

        save_item(fig, path)
        plt.close("all")

    def save_checkpoints(self, path="./tc_checkpoints.csv"):
        """
        Saves the checkpoints calculated during the training.

        Args:
            path (str): /path/to/save/tc.csv
        """
        save_item(self._checkpoints, path)

    @property
    def checkpoints(self):
        return self._checkpoints

    def __getattribute__(self, name):
        """
        Method to get access to all model attributes.

        """
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        """
        Method, mainly for the forward function of the wrapped model.
        """
        return self.model.__call__(*args, **kwargs)

    def __str__(self):
        return self.model.__str__()

## Tests
