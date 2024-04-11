import numpy as np
import torch
from torch.autograd import grad


class BaseTaylorAnalysis(object):
    """Class to wrap nn.Module for taylorcoefficient analysis. Base class for TaylorAnalysis. Use this class if you want to compute
    raw taylor coefficients or use your own plotting.
    """

    def __init__(self, eval_max_only: bool = True, apply_abs: bool = False):
        self._eval_max_only = eval_max_only
        self._apply_abs = apply_abs

    def _reduce(data):
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

    def first_order(self, x_data, **kwargs):
        """Compute all first order taylorcoefficients.

        Args:
            x_data (torch.tensor): X data of shape (batch, features).
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
        return self._reduce(gradients[0])

    def second_order(self, x_data, ind_i, **kwargs):
        """Compute second order taylorcoefficients according to ind_i and all other input variables.
        The model is first derivated according to the ind_i-th feature and second to all others.

        Args:
            x_data (torch.tensor): X data (batch, features).
            ind_i (int): Feature for the first derivative.
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
        return self._reduce(gradients)

    def third_order(self, x_data, ind_i, ind_j, **kwargs):
        """Compute third order taylorcoefficients according to ind_i, ind_j and all other input features.
        The model is derivated to the ind_i-th feature, the ind_j-th feature and third to all other features.

        Args:
            x_data (torch.tensor): X data (batch, features).
            ind_i (int): Feature for the first derivative.
            ind_j (int): Feature for the second derivative.
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
        return self._reduce(gradients)
