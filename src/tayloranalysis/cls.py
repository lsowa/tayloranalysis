import numpy as np
import torch

from collections import Counter
from math import factorial
from torch.autograd import grad
from abc import ABC
from typing import Tuple, List, Dict, Optional, Any, Union, Callable

##############################################
# Helper functions


def get_factorial_factors(*indices: List) -> float:
    """Function to compute the factorial factors for the taylorcoefficients: Prod_n^len(indices) 1/n!

    Returns:
        float: Factorial factors for the taylorcoefficients.
    """
    factor = 1.0
    counts = Counter(indices)
    for nth_variable, counts in counts.items():
        factor *= factorial(counts)
    return 1.0 / factor


def identity(x: torch.Tensor) -> torch.Tensor:
    """Identity function.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor.
    """
    return x


##############################################
# main class


class BaseTaylorAnalysis(ABC):
    """Class to wrap nn.Module for taylorcoefficient analysis. Base class for TaylorAnalysis. Use this class if you want to compute
    raw taylor coefficients or use your own plotting.
    """

    def _node_selection(
        self,
        pred: torch.Tensor,
        node: Optional[int],
        eval_max_node_only: bool,
    ) -> torch.Tensor:
        """Method to select the nodes for which the taylorcoefficients should be computed.

        Args:
            pred (torch.Tensor): X data of shape (batch, features).
            node (Optional[Int]): Node selection for evaluation. If None, all nodes are selected. If int, only the selected node is selected. If tuple, only the selected nodes are selected.
            eval_max_node_only (Bool): If True, only the node with the highest value is selected.

        Returns:
            torch.Tensor: Selected taylorcoefficients (batch, features)
        """
        # binary case skips everything
        if pred.dim() == 1 or pred.shape[1] == 1:
            # sum up everything
            return pred.sum()

        # first step: masking non max values if eval_max_node_only is set
        # and keeping only the output nodes with the highest value
        if eval_max_node_only:
            pred_view = pred.view(-1, pred.shape[-1])
            pred_cat = (
                (pred_view == pred_view.max(dim=1, keepdim=True)[0])
                .view_as(pred)
                .to(torch.float64)
            )
            pred = pred * pred_cat

        # second step: class selection
        # no selection is performed when node == "all"
        if isinstance(node, (int, tuple)):  # i.e. 0, (0, 1)
            pred = pred[:, node]

        # sum up everything
        pred = pred.sum()

        return pred

    @torch.enable_grad
    def _first_order(
        self,
        x_data: torch.Tensor,
        node: int,
        eval_max_node_only: bool,
        forward_dict: Dict,
        ind_i: int,
    ) -> torch.Tensor:
        """Method to compute the first order taylorcoefficients.

        Args:
            x_data (torch.Tensor): X data of shape (batch, features).
            node (Int): Node selection for evaluation.
            eval_max_node_only (Bool): If True, only the node with the highest value is selected.
            forward_dict (Dict): Dictionary with additional forward arguments.
            ind_i (Int): Index of the feature for which the taylorcoefficients should be computed.

        Returns:
            torch.Tensor: First order taylorcoefficients of shape (batch, features).
        """
        x_data.requires_grad = True
        self.zero_grad()
        x_data.grad = None
        pred = self(x_data, **forward_dict)
        pred = self._node_selection(pred, node, eval_max_node_only)
        # first order grads
        gradients = grad(pred, x_data)
        gradients = gradients[0][:, ind_i]
        return gradients

    @torch.enable_grad
    def _second_order(
        self,
        x_data: torch.Tensor,
        node: int,
        eval_max_node_only: bool,
        forward_dict: Dict,
        ind_i: int,
        ind_j: int,
    ) -> torch.Tensor:
        """Method to compute the second order taylorcoefficients.

        Args:
            x_data (torch.Tensor): X data of shape (batch, features).
            node (Int): Node selection for evaluation.
            eval_max_node_only (Bool): If True, only the node with the highest value is selected.
            forward_dict (Dict): Dictionary with additional forward arguments.
            ind_i (Int): First index of the feature for which the taylorcoefficients should be computed.
            ind_j (Int): Second index of the feature for which the taylorcoefficients should be computed.

        Returns:
            torch.Tensor: Second order taylorcoefficients of shape (batch, features).
        """
        x_data.requires_grad = True
        self.zero_grad()
        x_data.grad = None
        pred = self(x_data, **forward_dict)
        pred = self._node_selection(pred, node, eval_max_node_only)
        # first order gradients
        gradients = grad(pred, x_data, create_graph=True)
        gradients = gradients[0].sum(dim=0)
        # second order gradients
        gradients = grad(gradients[ind_i], x_data)
        gradients = gradients[0]
        # factor for second order taylor terms
        gradients *= get_factorial_factors(ind_i, ind_j)
        return gradients[:, ind_j]

    @torch.enable_grad
    def _third_order(
        self,
        x_data: torch.Tensor,
        node: int,
        eval_max_node_only: bool,
        forward_dict: Dict,
        ind_i: int,
        ind_j: int,
        ind_k: int,
    ) -> torch.Tensor:
        """Method to compute the third order taylorcoefficients.

        Args:
            x_data (torch.Tensor): X data of shape (batch, features).
            node (Int): Node selection for evaluation.
            eval_max_node_only (Bool): If True, only the node with the highest value is selected.
            forward_dict (Dict): Dictionary with additional forward arguments.
            ind_i (Int): First index of the feature for which the taylorcoefficients should be computed.
            ind_j (Int): Second index of the feature for which the taylorcoefficients should be computed.
            ind_k (Int): Third index of the feature for which the taylorcoefficients should be computed.

        Returns:
            torch.Tensor: Third order taylorcoefficients of shape (batch, features).
        """
        x_data.requires_grad = True
        self.zero_grad()
        x_data.grad = None
        pred = self(x_data, **forward_dict)
        pred = self._node_selection(pred, node, eval_max_node_only)
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
        gradients *= get_factorial_factors(ind_i, ind_j, ind_k)
        return gradients[:, ind_k]

    def _calculate_tc(
        self,
        x_data: torch.Tensor,
        node: int,
        eval_max_node_only: bool,
        forward_dict: Dict,
        *indices,
    ):
        """Method to calculate the taylorcoefficients based on the indices.

        Args:
            x_data (torch.Tensor): X data of shape (batch, features).
            node (Int): Node selection for evaluation.
            eval_max_node_only (Bool): If True, only the node with the highest value is selected.
            reduce_dict (Dict): Dictionary with additional reduce arguments.
            forward_dict (Dict): Dictionary with additional forward arguments.

        Raises:
            NotImplementedError: Only first, second and third order taylorcoefficients are supported.

        Returns:
            _type_: Output type is specified by the user defined reduce function.
        """

        functions = [self._first_order, self._second_order, self._third_order]
        try:
            out = functions[len(indices) - 1](
                x_data,
                node,
                eval_max_node_only,
                forward_dict,
                *indices,
            )
            return out
        except:
            raise NotImplementedError(
                "Only first, second and third order taylorcoefficients are supported."
            )

    def get_tc(
        self,
        x_data: torch.Tensor,
        ind_list: List[Tuple[int, ...]],
        node: Union[int, None] = None,
        eval_max_node_only: bool = True,
        reduce_func: Optional[Callable] = identity,
        reduce_dict: Union[None, Dict[str, Any]] = None,
        forward_dict: Union[None, Dict[str, Any]] = None,
    ) -> Dict[Tuple[int, ...], Any]:
        """Function to handle multiple indices and return the taylorcoefficients as a dictionary: to be used by the user.

        Args:
            x_data (torch.Tensor): X data of shape (batch, features).
            ind_list (List): List of indices for which the taylorcoefficients should be computed.
            node (Int, optional): Node selection for evaluation. Defaults to None.
            eval_max_node_only (Bool, optional): If True, only the node with the highest value is selected. Defaults to True.
            reduce_func (Callable, optional): Function to reduce the taylorcoefficients. Defaults to identity.
            reduce_dict (Dict, optional): Dictionary with additional reduce arguments. Defaults to {}.
            forward_dict (Dict, optional): Dictionary with additional forward arguments. Defaults to {}.

        Raises:
            ValueError: Ind_list must be a List of tuples!

        Returns:
            Dict: Dictionary with taylorcoefficients. Values are set by the user within the reduce function. Keys are the indices (tuple).
        """

        # set default values
        # dict can not directly used bc mutable properties might lead to unexpected behavior when calling function multiple times
        forward_dict = {} if forward_dict is None else forward_dict
        reduce_dict = {} if reduce_dict is None else reduce_dict

        # loop over all tc to compute
        output = {}
        for ind in ind_list:
            if not isinstance(ind, tuple):
                raise ValueError("Ind_list must be a list of tuples!")
            # get TCs
            out = self._calculate_tc(
                x_data, node, eval_max_node_only, forward_dict, *ind
            )
            # apply reduce function
            output[ind] = reduce_func(out, **reduce_dict)
        return output
