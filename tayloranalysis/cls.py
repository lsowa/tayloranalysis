import os
from copy import deepcopy
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
def save_item(item, path, prefix=None):
    if not isinstance(path, list):
        if prefix:
            _dir, _name = os.path.split(path)
            path = os.path.join(_dir, f"{prefix}_{_name}")
        if isinstance(item, matplotlib.figure.Figure):
            item.savefig(path, bbox_inches="tight")
        elif isinstance(item, dict):
            np.savez(path, **item)
        elif isinstance(item, pd.DataFrame):
            item.to_csv(path)
    if isinstance(path, list):
        for p in path:
            save_item(item=item, path=p, prefix=prefix)


class TaylorAnalysis(object):
    """
    Class to wrap nn.Module for taylorcoefficient analysis.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self._orders = {1: self._first_order, 2: self._second_order, 3: self._third_order}

    def _node_selection(self, pred, node=None):
        """In case of a multiclassification, selects a corresponding class (node) and, if
           necessary, masks individual entries (sets them to 0.0), if they are not
           maximal, i.e. not sorted into the corresponding class (self.eval_max_only).

        Args:
            pred (torch.tensor): X data of shape (batch, features).
            node (int, str, tuple[int]): class selection
        Returns:
            torch.tensor: First order taylorcoefficients (batch, features).
        """

        # binary case skips everything
        if pred.dim() == 1 or pred.shape[1] == 1:
            return pred

        # first step: masking non max values if self.eval_max_only is set
        # and keeping only the output nodes with the highest value
        if self.eval_max_only:
            pred_view = pred.view(-1, pred.shape[-1])
            pred_cat = (pred_view == pred_view.max(dim=1, keepdim=True)[0]).view_as(pred).to(torch.float64)
            pred = pred * pred_cat

        # second step: class selection
        # no selection is performed when node == "all"
        if isinstance(node, (int, tuple)):  # i.e. 0, (0, 1)
            pred = pred[:, node]

        return pred

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

    def _first_order(self, x_data, **kwargs):
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
        pred = self._node_selection(pred, **kwargs)
        pred = pred.sum()
        # first order grads
        gradients = grad(pred, x_data)
        return self._mean(gradients[0])

    def _second_order(self, x_data, ind_i, **kwargs):
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
        pred = self._node_selection(pred, **kwargs)
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

    def _third_order(self, x_data, ind_i, ind_j, **kwargs):
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
        pred = self._node_selection(pred, **kwargs)
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
        masked_factor = torch.tensor(masked_factor == ind_j, dtype=int) \
            + torch.tensor(masked_factor == ind_i, dtype=int) \
            + torch.tensor([ind_j == ind_i] * masked_factor.shape[0], dtype=int)
        masked_factor = (masked_factor == 1) * 2 + 1  # if variable pair is identical ..

        gradients *= masked_factor.to(gradients.device)
        return self._mean(gradients)

    def _get_derivatives(self, option, variable_idx, derivation_order, **kwargs):
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

    def _tc_checkpoint(
        self,
        x_data,
        epoch,
        dataframe,
        variable_mask,
        variable_idx,
        derivatives_for_calculation,
        **kwargs,
    ):
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
            mask = _mask(nplet)
            if any(mask):
                dataframe.loc[epoch, np.array(nplet)[mask]] = self._orders[len(item) + 1](x_data, *item, **kwargs)[variable_mask][mask]

    def tc_checkpoint(self, x_data, epoch):
        """
        Compute and save taylorcoefficients to plot and save them later.

        Args:
            x_data (torch.tensor): X data (batch, features).
            epoch (int): Current epoch.

        Returns:
            None
        """
        for node, dataframe in self._checkpoints.items():
            self._tc_checkpoint(
                x_data=x_data,
                epoch=epoch,
                dataframe=dataframe,
                variable_mask=self.variable_mask,
                variable_idx=self.variable_idx,
                derivatives_for_calculation=self.derivatives_for_calculation,
                node=node,
            )

    def setup_tc_checkpoints(
        self,
        number_of_variables_in_data,
        considered_variables_idx=None,
        variable_names=None,
        derivation_order=2,
        eval_nodes="all",
        eval_only_max_node=False,
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
            eval_nodes (int or (list, tuple)[int, tuple, str] or str):
                                        Compute Taylor Coefficients only based on the specified output node(s).
                                        If eval_nodes is set to "all" than all output nodes
                                        will be summed  and taken into account as one combined node.
                                        If a summation over two or more nodes is needed the eval_node
                                        have to be a list containing at least the tuple of nodes to be
                                        summed over.
            eval_only_max_node (bool): Compute Taylor Coefficients only based on the output node with
                                        the highest value. This step is done based on all output nodes.
        Returns:
            None
        """

        self.derivation_order = derivation_order
        self.variable_idx = considered_variables_idx or list(range(number_of_variables_in_data))
        self.variable_names = variable_names or [f"x_{i}" for i in range(number_of_variables_in_data)]
        self.variable_mask = np.array(self.variable_idx)
        self.eval_nodes = eval_nodes
        self.eval_max_only = eval_only_max_node

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
        _empty_dataframe = self._get_empty_checkpoints_dataframe(self.derivatives_for_dataframe)
        if eval_nodes == "all" or isinstance(eval_nodes, int):
            self._checkpoints = {key: deepcopy(_empty_dataframe) for key in [eval_nodes]}
        elif isinstance(eval_nodes, (list, tuple)):
            self._checkpoints = {key: deepcopy(_empty_dataframe) for key in eval_nodes}
        else:
            raise Exception("Provide 'eval_nodes' in form of an int, 'all' or a list of form i.e. [0, (0, 1), 'all']")

    def plot_checkpoints(self, path="./tc_training.pdf"):
        """
        Plot saved checkpoints.

        Args:
            path (str) or (list[str]): /path/to/save/plot.pdf or ["/path/to/save/plot.pdf", "/path/to/save/plot.png"]
        """
        for node, dataframe in self._checkpoints.items():
            fig_and_ax = [plt.subplots(1, 1, figsize=(10, 7)) for _ in range(self.derivation_order + 1)]
            fig, ax = tuple(zip(*fig_and_ax))  # 0: all, 1: first order, 2: second order...

            for column in dataframe.columns:
                _label = ",".join(np.array(self.variable_names)[np.array(column)])
                _label = f"$<t_{{{_label}}}>$"
                ax[0].plot(dataframe[column], label=_label, lw=lw)
                ax[len(column)].plot(dataframe[column], label=_label, lw=lw)

            for _ax in ax:
                _ax.legend(loc="upper left", bbox_to_anchor=(1.04, 1))
                _ax.set_xlabel("Epoch", loc="right", fontsize=13)
                _ax.set_ylabel("$<t_i>$", loc="top", fontsize=13)
                _ax.yaxis.set_tick_params(which="both", right=True, direction="in")
                _ax.xaxis.set_tick_params(which="both", top=True, direction="in")

            prefix = [""] + [f"order_{i+1}" for i in range(self.derivation_order)]
            prefix_node = "_".join(map(str, node)) if isinstance(node, tuple) else node
            prefix = [f"node_{prefix_node}_{pre}" for pre in prefix]
            for _fig, _pref in zip(fig, prefix):
                save_item(_fig, path, prefix=_pref)

            plt.close("all")

    def plot_taylor_coefficients(
        self,
        x_data,
        considered_variables_idx=None,
        variable_names=None,
        derivation_order=2,
        eval_nodes="all",
        eval_only_max_node=False,
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
        _derivatives_for_dataframe = self._get_derivatives(
            "dataframe",
            variable_idx=variable_idx,
            derivation_order=derivation_order,
        )
        _empty_dataframe = self._get_empty_checkpoints_dataframe(_derivatives_for_dataframe)
        if eval_nodes == "all" or isinstance(eval_nodes, int):
            _checkpoints = {key: deepcopy(_empty_dataframe) for key in [eval_nodes]}
        elif isinstance(eval_nodes, (list, tuple)):
            _checkpoints = {key: deepcopy(_empty_dataframe) for key in eval_nodes}
        else:
            raise Exception("Provide 'eval_nodes' in form of an int, 'all' or a list of form i.e. [0, (0, 1), 'all']")

        try:
            self._eval_max_only, self.eval_max_only = self.eval_max_only, eval_only_max_node  # copy
        except AttributeError:  # if not set before
            self.eval_max_only = eval_only_max_node

        for node, _dataframe in _checkpoints.items():
            self._tc_checkpoint(
                x_data=x_data,
                epoch=0,
                dataframe=_dataframe,
                variable_mask=variable_mask,
                variable_idx=variable_idx,
                derivatives_for_calculation=_derivatives_calculation,
                node=node,
            )

        try:
            self.eval_max_only = self._eval_max_only  # put it back
        except AttributeError:  # derefernce it if it was not was set previously
            del self.eval_max_only

        for node, _dataframe in _checkpoints.items():
            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
            xlabels = []

            for idx, column in enumerate(_dataframe.columns):
                _label = ",".join(np.array(self.variable_names)[np.array(column)])
                xlabels.append(f"$<t_{{{_label}}}>$")
                ax.plot(idx, _dataframe.loc[0][column], "+", color="black", markersize=10, markeredgewidth=markeredgewidth)

            ax.set_ylabel("$<t_i>$", loc="top")
            ax.set_xticks(list(range(idx + 1)))
            ax.set_xticklabels(xlabels, rotation=45, ha="right", rotation_mode="anchor")
            ax.grid(axis="x", alpha=0.25)
            plt.tight_layout()
            prefix = f'node_{"_".join(map(str, node)) if isinstance(node, tuple) else node}'

            save_item(fig, path, prefix=prefix)
            plt.close("all")

    def save_checkpoints(self, path="./tc_checkpoints.csv"):
        """
        Saves the checkpoints calculated during the training.

        Args:
            path (str): /path/to/save/tc.csv
        """
        for key, dataframe in self._checkpoints.items():
            save_item(
                item=dataframe,
                path=path,
                prefix=f'node_{"_".join(map(str, key)) if isinstance(key, tuple) else key}',
            )

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


# Tests
