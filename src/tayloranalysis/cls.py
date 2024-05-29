import torch

from collections import Counter
from math import factorial
from torch.autograd import grad
from typing import Tuple, List, Dict, Optional, Any, Union, Callable
from collections.abc import Sequence

##############################################
# Helpers


def _get_factorial_factors(*indices: int) -> float:
    """Function to compute the factorial factors for the taylorcoefficients: Prod_n^len(indices) 1/n!

    Returns:
        float: Factorial factors for the taylorcoefficients.
    """
    factor = 1.0
    counts = Counter(indices)
    for _, counts in counts.items():
        factor *= factorial(counts)
    return 1.0 / factor


def _get_summation_indices(shape: torch.Tensor.shape, drop_axis) -> Tuple[int, ...]:
    """Function to get the summation indices for the gradient.

    Args:
        shape (torch.Tensor.shape): Shape of the tensor.
        drop_axis (int): Axis to remove from the summation.

    Returns:
        Tuple[int, ...]: Indices that should be summed up.
    """

    idx = list(range(len(shape)))
    idx.pop(drop_axis)
    return tuple(idx)


def _get_slice(shape: torch.Tensor.shape, index: int, axis: int) -> Tuple[slice, ...]:
    """Function to get the slice for a tensor.

    Args:
        shape (torch.Tensor.shape): Shape of the tensor.
        index (int): Index that should be selected by the slice.
        axis (int): Axis of the given index.

    Returns:
        Tuple[slice, ...]: Slice for the tensor retrieval.
    """
    idx = [slice(None)] * len(shape)
    idx[axis] = index
    return tuple(idx)


def identity(x: torch.Tensor) -> torch.Tensor:
    """Identity function.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor.
    """
    return x


def _node_selection(
    pred: torch.Tensor,
    selected_output_node: int,
    eval_max_output_node_only: bool,
) -> torch.Tensor:
    """Method to select the nodes for which the taylorcoefficients should be computed.

    Args:
        pred (torch.Tensor): X data of shape (batch, features).
        output_node (Optional[Int]): Node selection for evaluation. If None, all nodes are selected. If int, only the selected node is selected. If tuple, only the selected nodes are selected.
        eval_max_selected_output_node_only (Bool): If True, only the node with the highest value is selected.

    Returns:
        torch.Tensor: Selected taylorcoefficients (batch, features)
    """
    # binary case skips everything
    if pred.dim() == 1 or pred.shape[1] == 1:
        # sum up everything
        return pred.sum()

    # first step: masking non max values if eval_max_output_node_only is set
    # and keeping only the output nodes with the highest value
    if eval_max_output_node_only:
        pred_view = pred.view(-1, pred.shape[-1])
        pred_cat = (
            (pred_view == pred_view.max(dim=1, keepdim=True)[0])
            .view_as(pred)
            .to(torch.float64)
        )
        pred = pred * pred_cat

    # second step: class selection
    # no selection is performed when selected_output_node == "all"
    if isinstance(selected_output_node, (int, tuple)):  # i.e. 0, (0, 1)
        pred = pred[:, selected_output_node]

    # sum up everything
    pred = pred.sum()
    return pred


class CustomForwardDict(dict):
    # custom dict wrapper to dynamically access the tctensor
    def __init__(
        self, forward_kwargs_tctensor_key: str, idx_to_tctensor: int, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self._key = forward_kwargs_tctensor_key
        self._idx = idx_to_tctensor

    @property
    def tctensor(self):
        """Property to easily access the tctensor.

        Raises:
            ValueError: If _idx is not an int
            NotImplementedError: Only list and tensor are supported as tctensor.

        Returns:
            torch.Tensor: tctensor.
        """
        # if the key is a list/tuple, return the element (tensor) at the given index
        if isinstance(self[self._key], Sequence):
            if not isinstance(self._idx, int):
                raise ValueError("Target index must be an integer!")
            return self[self._key][self._idx]
        # if the key is a tensor, return the tensor
        elif isinstance(self[self._key], torch.Tensor):
            return self[self._key]
        else:
            raise NotImplementedError("Only list and tensor are supported as tctensor.")


##############################################
# main class


class BaseTaylorAnalysis(object):
    """Class to wrap nn.Module for taylorcoefficient analysis. Base class for TaylorAnalysis. Use this class if you want to compute
    raw taylor coefficients or use your own plotting.
    """

    @torch.enable_grad()
    def _first_order(
        self,
        forward_kwargs: CustomForwardDict,
        tctensor_features_axis: int,
        pred: torch.Tensor,
        ind_i: int,
    ) -> torch.Tensor:
        """Method to compute the first order taylorcoefficients.

        Args:
            forward_kwargs (CustomForwardDict): (Custom) Dictionary with additional forward arguments
            pred (torch.Tensor): tensor with preselected predictions
            ind_i (Int): First index of the feature for which the taylorcoefficients should be computed.

        Returns:
            torch.Tensor: First order taylorcoefficients of shape (batch, features).
        """

        # first order grads
        gradients = grad(pred, forward_kwargs.tctensor)[0]
        return gradients[_get_slice(gradients.shape, ind_i, tctensor_features_axis)]

    @torch.enable_grad()
    def _second_order(
        self,
        forward_kwargs: CustomForwardDict,
        tctensor_features_axis: int,
        pred: torch.Tensor,
        ind_i: int,
        ind_j: int,
    ) -> torch.Tensor:
        """Method to compute the second order taylorcoefficients.

        Args:
            forward_kwargs (CustomForwardDict): (Custom) Dictionary with additional forward arguments
            pred (torch.Tensor): tensor with preselected predictions
            ind_i (Int): First index of the feature for which the taylorcoefficients should be computed.
            ind_j (Int): Second index of the feature for which the taylorcoefficients should be computed.

        Returns:
            torch.Tensor: Second order taylorcoefficients of shape (batch, features).
        """
        # first order gradients
        gradients = grad(pred, forward_kwargs.tctensor, create_graph=True)[0]
        gradients = gradients.sum(
            axis=_get_summation_indices(gradients.shape, tctensor_features_axis)
        )
        # second order gradients
        gradients = grad(gradients[ind_i], forward_kwargs.tctensor)[0]
        # factor for second order taylor terms
        gradients *= _get_factorial_factors(ind_i, ind_j)
        return gradients[_get_slice(gradients.shape, ind_j, tctensor_features_axis)]

    @torch.enable_grad()
    def _third_order(
        self,
        forward_kwargs: CustomForwardDict,
        tctensor_features_axis: int,
        pred: torch.Tensor,
        ind_i: int,
        ind_j: int,
        ind_k: int,
    ) -> torch.Tensor:
        """Method to compute the third order taylorcoefficients.

        Args:
            forward_kwargs (CustomForwardDict): (Custom) Dictionary with additional forward arguments
            pred (torch.Tensor): tensor with preselected predictions
            ind_i (Int): First index of the feature for which the taylorcoefficients should be computed.
            ind_j (Int): Second index of the feature for which the taylorcoefficients should be computed.
            ind_k (Int): Third index of the feature for which the taylorcoefficients should be computed.

        Returns:
            torch.Tensor: Third order taylorcoefficients of shape (batch, features).
        """
        # first order gradients
        gradients = grad(pred, forward_kwargs.tctensor, create_graph=True)[0]
        gradients = gradients.sum(
            axis=_get_summation_indices(gradients.shape, tctensor_features_axis)
        )
        # second order gradients
        gradients = grad(gradients[ind_i], forward_kwargs.tctensor, create_graph=True)[
            0
        ]
        gradients = gradients.sum(
            axis=_get_summation_indices(gradients.shape, tctensor_features_axis)
        )
        # third order gradients
        gradients = grad(gradients[ind_j], forward_kwargs.tctensor)[0]
        # factor for all third order taylor terms
        gradients *= _get_factorial_factors(ind_i, ind_j, ind_k)
        return gradients[_get_slice(gradients.shape, ind_k, tctensor_features_axis)]

    def _calculate_tc(
        self,
        forward_kwargs: CustomForwardDict,
        selected_output_node: int,
        eval_max_output_node_only: bool,
        tctensor_features_axis: int,
        selected_model_output_idx: int,
        *indices,
    ) -> torch.Tensor:
        """Method to calculate the taylorcoefficients based on the indices.

        Args:
            forward_kwargs (CustomForwardDict): (Custom) Dictionary with additional forward arguments
            selected_output_node (Int): Node selection for evaluation.
            eval_max_output_node_only (Bool): If True, only the node with the highest value is selected.
            tctensor_features_axis (int, optional): Dimension containing features in tctensor given in forward_kwargs. Defaults to -1.

        Raises:
            NotImplementedError: Only first, second and third order taylorcoefficients are supported.

        Returns:
            _type_: Output type is specified by the user defined reduce function.
        """

        # Make prediction
        forward_kwargs.tctensor.requires_grad = True
        self.zero_grad()
        forward_kwargs.tctensor.grad = None
        pred = self(**forward_kwargs)

        # select relevant predictions
        if isinstance(pred, Sequence):
            if selected_model_output_idx is None:
                raise ValueError(
                    "selected_model_output_idx must be set since model output is a sequence!"
                )
            pred = pred[selected_model_output_idx]
        pred = _node_selection(pred, selected_output_node, eval_max_output_node_only)

        # compute TCs
        functions = [self._first_order, self._second_order, self._third_order]
        try:
            return functions[len(indices) - 1](
                forward_kwargs,
                tctensor_features_axis,
                pred,
                *indices,
            )
        except KeyError:
            raise NotImplementedError(
                "Only first, second and third order taylorcoefficients are supported."
            )

    def get_tc(
        self,
        forward_kwargs_tctensor_key: str,
        forward_kwargs: Dict[str, Any],
        tc_idx_list: List[Tuple[int, ...]],
        *,
        selected_output_node: Optional[Union[int, Tuple[int]]] = None,
        eval_max_output_node_only: Optional[bool] = True,
        reduce_func: Optional[Callable] = identity,
        tctensor_features_axis: int = -1,
        additional_idx_to_tctensor: Optional[int] = None,
        selected_model_output_idx: Optional[int] = None,
    ) -> Dict[Tuple[int, ...], Any]:
        """Function to handle multiple indices and return the taylorcoefficients as a dictionary: to be used by the user.

        Args:
            forward_kwargs_tctensor_key (str): Key to input tensor in forward_kwargs. Based on this tensor the taylorcoefficients are computed.
            forward_kwargs (Union[None, Dict[str, Any]]): Dictionary with forward arguments
            tc_idx_list (List[Tuple[int, ...]]): List of indices for which the taylorcoefficients should be computed based on the tensor selected by forward_kwargs_tctensor_key (and additional_idx_to_tctensor).
            selected_output_node (Int, optional): Node selection for evaluation. Defaults to None.
            eval_max_output_node_only (Bool, optional): If True, only the node with the highest value is selected. Defaults to True.
            reduce_func (Callable, optional): Function to reduce the taylorcoefficients. Defaults to identity.
            tctensor_features_axis (int, optional): Dimension containing features in tctensor given in forward_kwargs. Defaults to -1.
            additional_idx_to_tctensor (int, optional): Index of the tctensor if forward_kwargs[forward_kwargs_tctensor_key] is a list. Defaults to None.
            selected_model_output_idx (int, optional): Index of the model output if its output is a sequence. Defaults to 0.
        Raises:
            ValueError: tc_idx_list must be a List of tuples!

        Returns:
            Dict: Dictionary with taylorcoefficients. Values are set by the user within the reduce function. Keys are the indices (tuple).
        """

        forward_kwargs = CustomForwardDict(
            forward_kwargs_tctensor_key, additional_idx_to_tctensor, forward_kwargs
        )

        assert isinstance(reduce_func, Callable), "Reduce function must be callable!"
        assert isinstance(
            selected_output_node, (int, tuple, type(None))
        ), "Node must be int, tuple or None!"

        # loop over all tc to compute
        output = {}
        for ind in tc_idx_list:
            if not isinstance(ind, tuple):
                raise ValueError("tc_idx_list must be a list of tuples!")
            # get TCs
            out = self._calculate_tc(
                forward_kwargs,
                selected_output_node,
                eval_max_output_node_only,
                tctensor_features_axis,
                selected_model_output_idx,
                *ind,
            )
            # apply reduce function
            output[ind] = reduce_func(out)
        return output
