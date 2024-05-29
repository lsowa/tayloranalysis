from types import MethodType
from typing import Union
from .cls import BaseTaylorAnalysis


def extend_model(
    model: Union[object, type],
):
    """function to extend a model with tayloranalysis functionality. This applies either to a class or an object.

    Args:
        model (Union[object, type]): model to be extended.
    Returns:
        Union[object, type]: extended model.
    """

    if isinstance(model, type):  # model is a class/not initialized
        # define wrapper class
        class WrappedModel(BaseTaylorAnalysis, model):
            def __init__(self, *args, **kwargs):
                BaseTaylorAnalysis.__init__(self)
                model.__init__(self, *args, **kwargs)

        WrappedModel.__name__ = model.__name__
        return WrappedModel
    else:  # model is an object/initialized
        # add all tayloranalysis.BaseTaylorAnalysis methods to the model

        model.get_tc = MethodType(BaseTaylorAnalysis.get_tc, model)

        model._first_order = MethodType(BaseTaylorAnalysis._first_order, model)
        model._second_order = MethodType(BaseTaylorAnalysis._second_order, model)
        model._third_order = MethodType(BaseTaylorAnalysis._third_order, model)
        model._calculate_tc = MethodType(BaseTaylorAnalysis._calculate_tc, model)

        return model
