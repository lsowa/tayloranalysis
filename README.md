# Taylorcoefficient Analysis for PyTorch modules
This is a pytorch implementation of the Paper
["Identifying the relevant dependencies of the neural network response on characteristics of the input space"](https://arxiv.org/abs/1803.08782)
(S. Wunsch, R. Friese, R. Wolf, G. Quast).

As explained in the paper, the method computes the taylorcoefficients of a taylored model function.

The analysis of taylorcoefficients is the optimal method to identify not only first order feature importance, but also higher order importance (i.e. the importance of combined features).

This module can be applied to any differentiable pytorch model.

## Installation

```
pip install git+https://gitlab.etp.kit.edu/lsowa/tayloranalysis.git
```

## Usage

Import tayloranalysis
```
import tayloranalysis as ta
```
Wrap either an already initialized PyTorch class instance or the class itself to extend it with the tayloranalysis functionality.
```
model = ...
model = extend_model(model)
```
Compute taylorcoefficients: for example $<t_{0}>$, $<t_{0,1}>$ for a given sample x_test.
Here you can also pass a reduction function to summarize the TCs, which are computed based on the `tctensor` `x_test`
```
combinations = [(0,), (0,1)]
x_test = torch.randn(#batch, #features)
forwards_kwargs = {"x": x_test, "more_inputs": misc}

tc_dict = model.get_tc(forward_kwargs_tctensor_key="x",
                        tc_idx_list=combinations, 
                        reduce_func=torch.mean,)
```
The output in this example is a dict containing the taylorcoefficients $<\mathrm{TC}_{0}>$, $<\mathrm{TC}_{0,1}>$.

## Maximal flexibility

This package is designed in a way to allow for maximal flexibility. While the reduction function has to be specified (e.g. mean, median, absolute values etc.) the visualization is up to the user. At this point you should have a look at our [example](example/example.ipynb).


## Authors
- [Lars Sowa](https://github.com/lsowa)
- [Artur Monsch](https://github.com/a-monsch)
