import string
from itertools import combinations_with_replacement as comb_wr
from itertools import permutations as perm
from itertools import product as prod
from types import MethodType
from typing import Any, Callable, Dict, Iterator, List, Set, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch as t


class ReductionError(Exception):
    pass


def nth_jacobian(
    func: Callable,
    X: t.Tensor,
    nth_order: int,
    selected_idx: List[List],
    **kwargs: Any,
) -> t.Tensor:
    if nth_order == 1:
        return t.autograd.functional.jacobian(func, X, **kwargs)[..., selected_idx[0]]
    else:
        return nth_jacobian(
            func=lambda _X: t.autograd.functional.jacobian(
                func,
                _X,
                create_graph=True,
                **kwargs,
            )[..., selected_idx[0]],
            X=X,
            nth_order=nth_order - 1,
            selected_idx=selected_idx[1:],
            **kwargs,
        )


def select_output_node(
    Y: t.Tensor,
    selected_idx: Union[int, Tuple[int], Any] = None,
    select_max_values: bool = False,
) -> t.Tensor:
    if Y.squeeze().dim() == 1:
        return Y

    if select_max_values:
        X_view = Y.view(-1, Y.shape[1])
        X_category = (X_view == X_view.max(dim=1, keepdim=True)[0]).view_as(Y).to(Y.dtype)
        Y = Y * X_category

    if isinstance(selected_idx, (int, tuple)):
        Y = Y[:, selected_idx]

    return Y


def tc_value_permutation(
    X: t.Tensor,
    tc_num: tuple,
    tc_map: Dict[int, int],
) -> npt.NDArray:

    if len(np.unique(tc_num)) == 1:  # is "diagonal"
        return X[(...,) + tuple(map(lambda it: tc_map[it], tc_num))]

    tc_num_idx = set(map(lambda it: tuple(map(lambda subit: tc_map[subit], it)), perm(tc_num, len(tc_num))))
    return np.hstack([X[(...,) + idx] for idx in tc_num_idx]).squeeze()


def output_dict(
    X: t.Tensor,
    tc_num: Iterator,
    tc_map: Dict[int, int],
    combine_permuations: bool,
    reduced_dim: bool,
) -> dict:
    def processed_tc(tc_num: tuple) -> npt.NDArray:

        if combine_permuations:
            tcs = tc_value_permutation(X, tc_num, tc_map)
            return tcs.sum() if reduced_dim else tcs

        return X[(...,) + tuple(map(lambda it: tc_map[it], tc_num))]

    return {idx: np.atleast_1d(processed_tc(idx)) for idx in set(tc_num)}


def adjust_tc_num(*args: Tuple[list, int]) -> Tuple[list, dict]:
    combined_tc_numbers = []
    for arg in args:
        combined_tc_numbers += [arg] if isinstance(arg, int) else list(arg)
    combined_tc_numbers = sorted(t.unique(t.as_tensor(combined_tc_numbers).to(int)).tolist())

    return (
        combined_tc_numbers,
        dict(map(reversed, enumerate(combined_tc_numbers))),  # type: ignore
    )


def einsum_reduction_string(n: int) -> str:
    unsummed = "".join([f"a{idx}" for idx in string.ascii_lowercase[1 : 1 + n]])
    summed = string.ascii_lowercase[: 1 + n]
    return f"{unsummed} -> {summed}"


class ReductionFunctions:
    @staticmethod
    def abs_mean(self: t.nn.Module, X: t.Tensor) -> npt.NDArray:
        return X.abs().mean(axis=0).cpu().detach().numpy()

    @staticmethod
    def mean(self: t.nn.Module, X: t.Tensor) -> npt.NDArray:
        return X.mean(axis=0).cpu().detach().numpy()

    @staticmethod
    def abs_median(self: t.nn.Module, X: t.Tensor) -> npt.NDArray:
        return X.abs().median(axis=0).cpu().detach().numpy()

    @staticmethod
    def median(self: t.nn.Module, X: t.Tensor) -> npt.NDArray:
        return X.median(axis=0).cpu().detach().numpy()

    @staticmethod
    def get_reduction_function(reduction: Union[str, Any, Callable] = "mean") -> Callable:
        if callable(reduction):
            return reduction

        elif reduction is None:
            return lambda self, X: X.cpu().detach().numpy()

        elif isinstance(reduction, str):
            if "mean" in reduction:
                return ReductionFunctions.abs_mean if "abs" in reduction else ReductionFunctions.mean

            elif "median" in reduction:
                return ReductionFunctions.abs_median if "abs" in reduction else ReductionFunctions.median

        raise NotImplementedError(
            "Provide a reduction of form 'abs mean', 'mean', 'abs median', 'median' or " "as a Callable returning a numpy array"
        )


class TaylorCoefficientCalculation:

    # All methods will recieve tc_ prefix in patched model class

    @staticmethod
    def first_order(
        self: t.nn.Module,
        X: t.Tensor,
        i: Union[int, list],
        *,
        chunk_size: int = 10,
        vectorize: bool = True,
        strategy: str = "reverse-mode",
        **kwargs: Any,
    ) -> npt.NDArray:
        return self.tc_nth_order(
            X,
            i,
            combine_permutations=False,
            chunk_size=chunk_size,
            vectorize=vectorize,
            strategy=strategy,
            **kwargs,
        )

    @staticmethod
    def second_order(
        self: t.nn.Module,
        X: t.Tensor,
        i: Union[int, list],
        j: Union[int, list],
        *,
        combine_permutations: bool = True,
        chunk_size: int = 10,
        vectorize: bool = True,
        strategy: str = "reverse-mode",
        **kwargs: Any,
    ) -> npt.NDArray:
        return self.tc_nth_order(
            X,
            i,
            j,
            combine_permutations=combine_permutations,
            chunk_size=chunk_size,
            vectorize=vectorize,
            strategy=strategy,
            **kwargs,
        )

    @staticmethod
    def third_order(
        self: t.nn.Module,
        X: t.Tensor,
        i: Union[int, list],
        j: Union[int, list],
        k: Union[int, list],
        *,
        combine_permutations: bool = True,
        chunk_size: int = 10,
        vectorize: bool = True,
        strategy: str = "reverse-mode",
        **kwargs: Any,
    ) -> npt.NDArray:
        return self.tc_nth_order(
            X,
            i,
            j,
            k,
            combine_permutations=combine_permutations,
            chunk_size=chunk_size,
            vectorize=vectorize,
            strategy=strategy,
            **kwargs,
        )

    @staticmethod
    def nth_order(
        self: t.nn.Module,
        X: t.Tensor,
        *tc_idx: Tuple[int, list],
        combine_permutations: bool = True,
        chunk_size: int = 10,
        vectorize: bool = True,
        strategy: str = "reverse-mode",
        **kwargs: Any,
    ) -> Dict[tuple, npt.NDArray]:

        idx, idx_map = adjust_tc_num(*tc_idx)  # type: ignore
        n_chunks = int(np.ceil(X.shape[0] / chunk_size))
        reduction_string = einsum_reduction_string(len(tc_idx))
        denominator = np.math.factorial(len(tc_idx))  # type: ignore

        X.requires_grad = True
        self.zero_grad()
        X.grad = None

        def nth_order_derivative(X: t.Tensor) -> t.Tensor:
            return t.einsum(
                reduction_string,  # "abacad... -> abcd..."
                nth_jacobian(
                    func=lambda _X: select_output_node(self(_X), **kwargs).sum(),
                    X=X,
                    nth_order=len(tc_idx),
                    selected_idx=len(tc_idx) * [idx],
                    strategy=strategy,
                    vectorize=vectorize,
                ),
            )

        gradients = t.vstack([nth_order_derivative(_X) / denominator for _X in X.chunk(n_chunks)])

        try:
            reduced_gradients = self.tc_reduce(gradients)
            reduced_dim = bool(len(gradients.shape) - len(reduced_gradients.shape))
        except Exception as exception:
            raise ReductionError(f"TCs could not be reduced by the provided reduction function, raising {exception}")

        tc_num = comb_wr(idx, len(tc_idx)) if combine_permutations else prod(*[idx] * len(tc_idx))

        return output_dict(
            X=reduced_gradients,
            tc_num=tc_num,
            tc_map=idx_map,
            combine_permuations=combine_permutations,
            reduced_dim=reduced_dim,
        )


def model_extension(
    model: t.nn.Module,
    reduction: Union[str, Any, Callable] = "abs mean",
) -> t.nn.Module:

    model.tc_reduce = MethodType(ReductionFunctions.get_reduction_function(reduction=reduction), model)

    model.tc_nth_order = MethodType(TaylorCoefficientCalculation.nth_order, model)

    model.tc_first_order = MethodType(TaylorCoefficientCalculation.first_order, model)
    model.tc_second_order = MethodType(TaylorCoefficientCalculation.second_order, model)
    model.tc_third_order = MethodType(TaylorCoefficientCalculation.third_order, model)
