from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd
import torch
from torch.autograd import grad

from .plots import checkpoints_from_single_values, taylor_coefficients_from_single_values
from .utils import Summarization, get_external_representation, save_item


class TaylorAnalysis(object):
    """
    Class to wrap nn.Module for taylorcoefficient analysis.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self._orders = {1: self._first_order, 2: self._second_order, 3: self._third_order}

        self.summarization_function = Summarization.abs_mean
        self.checkpoint_plot_function = checkpoints_from_single_values
        self.tc_plot_function = taylor_coefficients_from_single_values

        self._apply_abs = "abs" in self.summarization_function.__name__  # TODO: Find a context where this could be set?

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
            # sum up everything
            return pred.sum()

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

        # sum up everything
        pred = pred.sum()

        return pred

    def _summarization(self, data):
        """Passtrough the summarization_function provided by user.

        Args:
            data (torch.tensor): tensor with taylorcoefficients of shape (batch, features)

        Returns:
            numpy.array: Array means of taylorcoefficients.
        """
        return self.summarization_function(data)

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
        # first order grads
        gradients = grad(pred, x_data)
        return self._summarization(gradients[0])

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
        return self._summarization(gradients)

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

        gradients *= masked_factor.to(gradients.device)
        return self._summarization(gradients)

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
        _df = _df.astype(object)
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

        def _nplet(*__x):
            return [(idx,) for idx in variable_idx] if __x is None else [(idx, *__x) for idx in variable_idx]

        def _mask(__nplet):
            return pd.Series(__nplet).isin(dataframe.columns).to_numpy()

        for item in derivatives_for_calculation:
            nplet = _nplet(*item)
            mask = _mask(nplet)
            if any(mask):
                dataframe.loc[epoch, np.array(nplet)[mask]] = list(
                    self._orders[len(item) + 1](x_data, *item, **kwargs)[variable_mask][mask]
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
            self._tc_points = {key: deepcopy(_empty_dataframe) for key in [eval_nodes]}
        elif isinstance(eval_nodes, (list, tuple)):
            self._tc_points = {key: deepcopy(_empty_dataframe) for key in eval_nodes}
        else:
            raise Exception("Provide 'eval_nodes' in form of an int, 'all' or a list of form i.e. [0, (0, 1), 'all']")

    def tc_checkpoint(self, x_data, epoch):
        """
        Compute and save taylorcoefficients to plot and save them later.

        Args:
            x_data (torch.tensor): X data (batch, features).
            epoch (int): Current epoch.

        Returns:
            None
        """
        for node, dataframe in self._tc_points.items():
            self._tc_checkpoint(
                x_data=x_data,
                epoch=epoch,
                dataframe=dataframe,
                variable_mask=self.variable_mask,
                variable_idx=self.variable_idx,
                derivatives_for_calculation=self.derivatives_for_calculation,
                node=node,
            )

    def calculate_tc(
        self,
        x_data,
        considered_variables_idx=None,
        derivation_order=2,
        eval_nodes="all",
        eval_only_max_node=False,
        **kwargs,
    ):
        """
        calculate taylorcoefficients for current weights of the model.

        Args:
            x_data (torch.tensor): X data of shape (batch, features).
            considered_variables_idx (list[int]): Contains the indices of variables according to which
                                                  the derivation is required. All variables are
                                                  considered, unless explicitly stated otherwise.
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

        """
        self._variable_idx = considered_variables_idx or list(range(x_data.shape[1]))
        self._variable_mask = np.array(self._variable_idx)

        _derivatives_calculation = self._get_derivatives(
            "calculation",
            variable_idx=self._variable_idx,
            derivation_order=derivation_order,
        )
        _derivatives_for_dataframe = self._get_derivatives(
            "dataframe",
            variable_idx=self._variable_idx,
            derivation_order=derivation_order,
        )
        _empty_dataframe = self._get_empty_checkpoints_dataframe(_derivatives_for_dataframe)
        if eval_nodes == "all" or isinstance(eval_nodes, int):
            self._tc_point = {key: deepcopy(_empty_dataframe) for key in [eval_nodes]}
        elif isinstance(eval_nodes, (list, tuple)):
            self._tc_point = {key: deepcopy(_empty_dataframe) for key in eval_nodes}
        else:
            raise Exception("Provide 'eval_nodes' in form of an int, 'all' or a list of form i.e. [0, (0, 1), 'all']")

        try:
            self._eval_max_only, self.eval_max_only = self.eval_max_only, eval_only_max_node  # copy
        except AttributeError:  # if not set before
            self.eval_max_only = eval_only_max_node

        for node, _dataframe in self._tc_point.items():
            self._tc_checkpoint(
                x_data=x_data,
                epoch=0,
                dataframe=_dataframe,
                variable_mask=self._variable_mask,
                variable_idx=self._variable_idx,
                derivatives_for_calculation=_derivatives_calculation,
                node=node,
            )

        try:
            self.eval_max_only = self._eval_max_only  # put it back
        except AttributeError:  # derefernce it if it was not was set previously
            del self.eval_max_only

    def plot_taylor_coefficients(self, *args, **kwargs):
        """
        Plot taylorcoefficients for current weights of the model.
        calculate_tc if x_data is provided. Otherwise calculate_tc
        before calling this function

        Args and Kwargs are passed to defined tc_plot_function.
        """

        if "x_data" in kwargs:
            self.calculate_tc(x_data=kwargs.pop("x_data"), **kwargs)
            for item in [
                "considered_variables_idx",
                "derivation_order",
                "eval_nodes",
                "eval_only_max_node",
            ]:
                try:
                    kwargs.pop(item)
                except KeyError:
                    pass
        if not hasattr(self, "_tc_point"):
            raise AttributeError("Run 'calculate_tc' first or provide kwargs for 'calculate_tc' inplace.")
        self.tc_plot_function(self, *args, **kwargs)

    def plot_checkpoints(self, *args, **kwargs):
        """
        Plot saved checkpoints with a given function.

        Args and Kwargs are passed to defined checkpoint_plot_function.
        """
        self.checkpoint_plot_function(self, *args, **kwargs)

    def save_tc(
        self,
        path=None,
        path_tc_points=None,
        path_tc_point=None
    ):
        """
        Saves the checkpoints calculated during the training.

        Args:
            path (str): /path/to/save/tc.csv to save all calculated tcs
            tc_points_path (str): /path/to/save/tc.csv to save tc from checkpoints
            tc_points_path (str): /path/to/save/tc.csv to save tc from calculate_tc
        """
        if (path or path_tc_points) and hasattr(self, "_tc_points"):
            for key, dataframe in self.tc_points.items():
                save_item(
                    item=dataframe,
                    path=path or path_tc_points,
                    prefix=f'training_node_{"_".join(map(str, key)) if isinstance(key, tuple) else key}',
                )
        if (path or path_tc_point) and hasattr(self, "_tc_point"):
            for key, dataframe in self.tc_point.items():
                save_item(
                    item=dataframe,
                    path=path or path_tc_point,
                    prefix=f'testing_node_{"_".join(map(str, key)) if isinstance(key, tuple) else key}',
                )

    @property
    def tc_points(self):

        current_tc_points_id = id(self._tc_points)

        try:
            if current_tc_points_id != self._previous_tc_points_id:
                self._tc_points_external_representation = get_external_representation(
                    tc_collection_dict=self._tc_points,
                    summarization_function=self.summarization_function,
                )
                self._previous_tc_points_id = current_tc_points_id
        except AttributeError:
            self._previous_tc_points_id = current_tc_points_id
            self._tc_points_external_representation = get_external_representation(
                tc_collection_dict=self._tc_points,
                summarization_function=self.summarization_function,
            )

        return self._tc_points_external_representation

    @property
    def tc_point(self):

        current_tc_point_id = id(self._tc_point)

        try:
            if current_tc_point_id != self._previous_tc_point_id:
                self._tc_point_external_representation = get_external_representation(
                    tc_collection_dict=self._tc_point,
                    summarization_function=self.summarization_function,
                    convert_to_single_point=True,
                )
                self._previous_tc_point_id = current_tc_point_id
        except AttributeError:
            self._previous_tc_point_id = current_tc_point_id
            self._tc_point_external_representation = get_external_representation(
                tc_collection_dict=self._tc_point,
                summarization_function=self.summarization_function,
                convert_to_single_point=True,
            )

        return self._tc_point_external_representation

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
