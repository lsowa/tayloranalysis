import abc
from types import MethodType
from typing import Any, Callable, Union

import tayloranalysis


def extend_model(model: Union[object, type], tc_reduce_function: Callable):
    if isinstance(model, type):  # model is not initiated

        class WrappedModel(model, tayloranalysis.TaylorCoefficientBase):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def tc_reduce(self, *args: Any, **kwargs: Any) -> Any:
                return tc_reduce_function(self, *args, **kwargs)

        WrappedModel.__name__ = model.__name__

        return WrappedModel

    else:  # model is allread initiated
        model.tc_reduce = MethodType(tc_reduce_function, model)
        model.nth_order_tc = MethodType(
            tayloranalysis.TaylorCoefficientBase.nth_order_tc, model
        )

        return model
