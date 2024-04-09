import abc
from typing import Any, Union, List, Tuple, Dict
import torch


class TaylorCoefficientBase(abc.ABC):
    @abc.abstractmethod
    def tc_reduce(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def nth_order_tc(
        self,
        X: torch.Tensor,
        *args: Union[int, List[Tuple[int, ...]]],
        **kwargs: Any,
    ) -> Dict[Tuple[int, ...], torch.Tensor]:
        if all(isinstance(it, (list, tuple)) for it in args):
            return {k: self.tc_reduce(X, *k, **kwargs) for k in args}
        elif all(isinstance(it, int) for it in args):
            return {args: self.tc_reduce(X, *args, **kwargs)}
        else:
            raise ValueError("Invalid input")
