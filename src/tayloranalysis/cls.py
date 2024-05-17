import torch

from collections import Counter
from math import factorial
from torch.autograd import grad
from typing import Tuple, List, Dict, Optional, Any, Union, Callable
from collections.abc import Sequence

##############################################
# Helpers


def get_factorial_factors(*indices: int) -> float:
    """Function to compute the factorial factors for the taylorcoefficients: Prod_n^len(indices) 1/n!

    Returns:
        float: Factorial factors for the taylorcoefficients.
    """
    factor = 1.0
    counts = Counter(indices)
    for _, counts in counts.items():
        factor *= factorial(counts)
    return 1.0 / factor


def get_summation_indices(shape: torch.Tensor.shape, drop_axis) -> Tuple[int, ...]:
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


def get_slice(shape: torch.Tensor.shape, index: int, axis: int) -> Tuple[slice, ...]:
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


class CustomForwardDict(dict):
    # custom dict wrapper to dynamically access the derivation target
    def __init__(self, target_key: str, target_idx: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._key = target_key
        self._idx = target_idx

    @property
    def deriv_target(self):
        """Property to easily access the derivation target.

        Raises:
            ValueError: If _idx is not an int
            NotImplementedError: Only list and tensor are supported as derivation targets.

        Returns:
            torch.Tensor: Derivation target.
        """
        # if the key is a list/tuple, return the element (tensor) at the given index
        if isinstance(self.__getitem__(self._key), Sequence):
            if not isinstance(self._idx, int):
                raise ValueError("Target index must be an integer!")
            return self.__getitem__(self._key)[self._idx]
        # if the key is a tensor, return the tensor
        elif isinstance(self.__getitem__(self._key), torch.Tensor):
            return self.__getitem__(self._key)
        else:
            raise NotImplementedError(
                "Only list and tensor are supported as derivation targets."
            )


def _node_selection(
    pred: torch.Tensor,
    output_node: int,
    eval_max_output_node_only: bool,
) -> torch.Tensor:
    """Method to select the nodes for which the taylorcoefficients should be computed.

    Args:
        pred (torch.Tensor): X data of shape (batch, features).
        output_node (Optional[Int]): Node selection for evaluation. If None, all nodes are selected. If int, only the selected node is selected. If tuple, only the selected nodes are selected.
        eval_max_output_node_only (Bool): If True, only the node with the highest value is selected.

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
    # no selection is performed when output_node == "all"
    if isinstance(output_node, (int, tuple)):  # i.e. 0, (0, 1)
        pred = pred[:, output_node]

    # sum up everything
    pred = pred.sum()
    return pred


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
        features_axis: int,
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
        gradients = grad(pred, forward_kwargs.deriv_target)[0]
        return gradients[get_slice(gradients.shape, ind_i, features_axis)]

    @torch.enable_grad()
    def _second_order(
        self,
        forward_kwargs: CustomForwardDict,
        features_axis: int,
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
        gradients = grad(pred, forward_kwargs.deriv_target, create_graph=True)[0]
        gradients = gradients.sum(
            axis=get_summation_indices(gradients.shape, features_axis)
        )
        # second order gradients
        gradients = grad(gradients[ind_i], forward_kwargs.deriv_target)[0]
        # factor for second order taylor terms
        gradients *= get_factorial_factors(ind_i, ind_j)
        return gradients[get_slice(gradients.shape, ind_j, features_axis)]

    @torch.enable_grad()
    def _third_order(
        self,
        forward_kwargs: CustomForwardDict,
        features_axis: int,
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
        gradients = grad(pred, forward_kwargs.deriv_target, create_graph=True)[0]
        gradients = gradients.sum(
            axis=get_summation_indices(gradients.shape, features_axis)
        )
        # second order gradients
        gradients = grad(
            gradients[ind_i], forward_kwargs.deriv_target, create_graph=True
        )[0]
        gradients = gradients.sum(
            axis=get_summation_indices(gradients.shape, features_axis)
        )
        # third order gradients
        gradients = grad(gradients[ind_j], forward_kwargs.deriv_target)[0]
        # factor for all third order taylor terms
        gradients *= get_factorial_factors(ind_i, ind_j, ind_k)
        return gradients[get_slice(gradients.shape, ind_k, features_axis)]

    def _calculate_tc(
        self,
        forward_kwargs: CustomForwardDict,
        output_node: int,
        eval_max_output_node_only: bool,
        features_axis: int,
        keep_model_output_idx: int,
        *indices,
    ) -> torch.Tensor:
        """Method to calculate the taylorcoefficients based on the indices.

        Args:
            forward_kwargs (CustomForwardDict): (Custom) Dictionary with additional forward arguments
            output_node (Int): Node selection for evaluation.
            eval_max_output_node_only (Bool): If True, only the node with the highest value is selected.
            features_axis (int, optional): Dimension containing features in tensor forward_kwargs.deriv_target. Defaults to -1.

        Raises:
            NotImplementedError: Only first, second and third order taylorcoefficients are supported.

        Returns:
            _type_: Output type is specified by the user defined reduce function.
        """

        # Make prediction
        forward_kwargs.deriv_target.requires_grad = True
        self.zero_grad()
        forward_kwargs.deriv_target.grad = None
        pred = self(**forward_kwargs)

        # select relevant predictions
        if isinstance(pred, Sequence):
            if keep_model_output_idx is None:
                raise ValueError(
                    "keep_model_output_idx must be set since model output is a sequence!"
                )
            pred = pred[keep_model_output_idx]
        pred = _node_selection(pred, output_node, eval_max_output_node_only)

        # compute TCs
        functions = [self._first_order, self._second_order, self._third_order]
        try:
            return functions[len(indices) - 1](
                forward_kwargs,
                features_axis,
                pred,
                *indices,
            )
        except KeyError:
            raise NotImplementedError(
                "Only first, second and third order taylorcoefficients are supported."
            )

    def get_tc(
        self,
        target_key: str,
        forward_kwargs: Dict[str, Any],
        idx_list: List[Tuple[int, ...]],
        *,
        output_node: Optional[Union[int, Tuple[int], None]] = None,
        eval_max_output_node_only: Optional[bool] = True,
        reduce_func: Optional[Callable] = identity,
        features_axis: int = -1,
        output_idx: Union[int, None] = None,
        keep_model_output_idx: Union[int, None] = None,
    ) -> Dict[Tuple[int, ...], Any]:
        """Function to handle multiple indices and return the taylorcoefficients as a dictionary: to be used by the user.

        Args:
            target_key (str): Key to input tensor in forward_kwargs. Based on this tensor the taylorcoefficients are computed.
            forward_kwargs (Union[None, Dict[str, Any]]): Dictionary with forward arguments
            idx_list (List[Tuple[int, ...]]): List of indices for which the taylorcoefficients should be computed.
            output_node (Int, optional): Node selection for evaluation. Defaults to None.
            eval_max_output_node_only (Bool, optional): If True, only the node with the highest value is selected. Defaults to True.
            reduce_func (Callable, optional): Function to reduce the taylorcoefficients. Defaults to identity.
            features_axis (int, optional): Dimension containing features in tensor forward_kwargs.deriv_target. Defaults to -1.
            output_idx (Union[int, None], optional): Index of the target tensor if forward_kwargs[target_key] is a list. Defaults to None.
            keep_model_output_idx (int, optional): Index of the model output if its output is a sequence. Defaults to 0.
        Raises:
            ValueError: idx_list must be a List of tuples!

        Returns:
            Dict: Dictionary with taylorcoefficients. Values are set by the user within the reduce function. Keys are the indices (tuple).
        """

        forward_kwargs = CustomForwardDict(target_key, output_idx, forward_kwargs)

        assert isinstance(reduce_func, Callable), "Reduce function must be callable!"
        assert isinstance(
            output_node, (int, tuple, type(None))
        ), "Node must be int, tuple or None!"

        # loop over all tc to compute
        output = {}
        for ind in idx_list:
            if not isinstance(ind, tuple):
                raise ValueError("idx_list must be a list of tuples!")
            # get TCs
            out = self._calculate_tc(
                forward_kwargs,
                output_node,
                eval_max_output_node_only,
                features_axis,
                keep_model_output_idx,
                *ind,
            )
            # apply reduce function
            output[ind] = reduce_func(out)
        return output
