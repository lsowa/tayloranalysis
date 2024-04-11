import numpy as np
import torch
from torch.autograd import grad
import pandas as pd


class BaseTaylorAnalysis(object):
    """Class to wrap nn.Module for taylorcoefficient analysis. Base class for TaylorAnalysis. Use this class if you want to compute
    raw taylor coefficients or use your own plotting.
    """

    def __init__(self, eval_max_only: bool = True, apply_abs: bool = False):
        """Initializes the BaseTaylorAnalysis class.
        Args:
            eval_max_only (bool, optional): Compute Taylor Coefficients only based on the output node with
                                        the highest value. This step is done based on all output nodes. Defaults to True.
            apply_abs (bool, optional): Specifies if the TCs should be computed as absolute values. Defaults to False.
        """
        self._eval_max_only = eval_max_only
        self._apply_abs = apply_abs

    def _reduce(x_data):
        """reduce method to be implemented by the user.

        Args:
            x_data (_type_): x_data of shape (batch, features).
        """
        pass

    def _node_selection(self, pred, node=None):
        """In case of a multiclassification, selects a corresponding class (node) and, if
           necessary, masks individual entries (sets them to 0.0), if they are not
           maximal, i.e. not sorted into the corresponding class (self._eval_max_only).

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

        # first step: masking non max values if self._eval_max_only is set
        # and keeping only the output nodes with the highest value
        if self._eval_max_only:
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

    def _first_order(self, x_data, ind_i, **kwargs):
        """Compute all first order taylorcoefficients.

        Args:
            x_data (torch.tensor): X data of shape (batch, features).
            ind_i (int): Feature for the first derivative.
            node (int, str, tuple[int]): class selection

        Returns:
            torch.tensor: First order taylorcoefficients (batch, feature).
        """
        x_data.requires_grad = True
        self.zero_grad()
        x_data.grad = None
        pred = self.__call__(x_data)
        pred = self._node_selection(pred, **kwargs)
        # first order grads
        gradients = grad(pred, x_data)
        gradients = gradients[0][:, ind_i]
        return self._reduce(gradients)

    def _second_order(self, x_data, ind_i, ind_j, **kwargs):
        """Compute second order taylorcoefficients according to ind_i and all other input variables.
        The model is first derivated according to the ind_i-th feature and second to all others.

        Args:
            x_data (torch.tensor): X data (batch, features).
            ind_i (int): Feature for the first derivative.
            ind_j (int): Feature for the second derivative.
            node (int, str, tuple[int]): class selection

        Returns:
            torch.tensor: Second order derivatives according to ind_i and all other input variables (batch, feature).
        """
        x_data.requires_grad = True
        self.zero_grad()
        x_data.grad = None
        pred = self.__call__(x_data)
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
        return self._reduce(gradients[:, ind_j])

    def _third_order(self, x_data, ind_i, ind_j, ind_k, **kwargs):
        """Compute third order taylorcoefficients according to ind_i, ind_j and all other input features.
        The model is derivated to the ind_i-th feature, the ind_j-th feature and third to all other features.

        Args:
            x_data (torch.tensor): X data (batch, features).
            ind_i (int): Feature for the first derivative.
            ind_j (int): Feature for the second derivative.
            ind_k (int): Feature for the third derivative.
            node (int, str, tuple[int]): class selection

        Returns:
            torch.tensor: Third order derivatives according to ind_i, ind_j and all other input features (batch, feature).
        """
        x_data.requires_grad = True
        self.zero_grad()
        x_data.grad = None
        pred = self.__call__(x_data)
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
        return self._reduce(gradients[:, ind_k])

    def _calculate_tc(self, x_data, *indices, **kwargs):
        """function to calculate the taylorcoefficients based on the given indices.

        Args:
            x_data (torch.tensor): X data (batch, features).

        Returns:
            torch.tensor: reduced taylorcoefficient for given indices.
        """
        if len(indices) == 1:
            return self._first_order(x_data, *indices, **kwargs)
        elif len(indices) == 2:
            return self._second_order(x_data, *indices, **kwargs)
        elif len(indices) == 3:
            return self._third_order(x_data, *indices, **kwargs)
        else:
            raise ValueError(
                "Only first, second and third order taylorcoefficients are supported."
            )

    def get_tc(self, x_data, ind_list, feature_names=None, **kwargs):
        """function to handle multiple indices and return the taylorcoefficients as a dictionary: to be used by the user.

        Args:
            x_data (torch.tensor): X data (batch, features).
            ind_list (list of lists): list of indices to compute the taylorcoefficients for.
            feature_names (list, optional): list of feature names to create dictionary keys. Defaults to None.

        Returns:
            dict: dictionary with the taylorcoefficients for the given indices.
        """
        out = {}
        for ind in ind_list:
            if isinstance(ind, int):
                ind = [ind]

            # create column name
            if feature_names is not None:
                col_name = ",".join([feature_names[i] for i in ind])
            else:
                col_name = str(ind).replace(" ", "").replace("[", "").replace("]", "")

            out[col_name] = float(self._calculate_tc(x_data, *ind, **kwargs))
        return out
