from types import MethodType
from typing import Any, Callable, Union

import tayloranalysis


def extend_model(
    model: Union[object, type],
    reduce_function: Callable,
    eval_max_only: bool = True,
    apply_abs: bool = False,
):
    """function to extend a model with tayloranalysis functionality. This applies either to a class or an object.

    Args:
        model (Union[object, type]): model to be extended.
        reduce_function (Callable): function to reduce the taylorcoefficients to
            one value. This function has to be defined by the user.
        eval_max_only (bool, optional): Specifies if the TCs should be computed based on
            the highest value. This step is done based on all output nodes. Defaults to True.
        apply_abs (bool, optional): Specifies if the TCs should be computed as absolute values. Defaults to False.
    Returns:
        Union[object, type]: extended model.
    """

    if isinstance(model, type):  # model is a class/not initialized
        # define wrapper class
        class WrappedModel(model, tayloranalysis.BaseTaylorAnalysis):
            def __init__(self, *args, **kwargs):
                model.__init__(self, *args, **kwargs)
                tayloranalysis.BaseTaylorAnalysis.__init__(
                    self, eval_max_only=eval_max_only, apply_abs=apply_abs
                )

            def _reduce(self, *args, **kwargs):
                return reduce_function(*args, **kwargs)

        WrappedModel.__name__ = model.__name__
        return WrappedModel
    else:  # model is an object/initialized
        # add all tayloranalysis.BaseTaylorAnalysis methods to the model
        for name, value in vars(tayloranalysis.BaseTaylorAnalysis).items():
            if not name.startswith("__"):
                setattr(model, name, value.__get__(model))

        # wrap the reduce function to add self as first argument
        def _reduce_function(self, *args, **kwargs):
            return reduce_function(*args, **kwargs)

        model._reduce = MethodType(_reduce_function, model)
        model._eval_max_only = eval_max_only
        model._apply_abs = apply_abs
        return model
