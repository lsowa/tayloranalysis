# Taylorcoefficient Analysis
This is a pytorch implementation of the Paper
["Identifying the relevant dependencies of the neural network response on characteristics of the input space"](https://arxiv.org/abs/1803.08782)
(S. Wunsch, R. Friese, R. Wolf, G. Quast).

As explained in the paper, the method computes the taylorcoefficients of a taylored model function. To allow for more flexibility, this package requieres manual specification of the reduction function.

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
Wrap either an already initialized PyTorch class instance or the class itself to extend it with the tayloranalysis functionality and choose the reduction function to be used.
```
model = ...
model = extend_model(model, reduce_function=torch.mean)
```
Compute taylorcoefficients: for example $<t_{0}>$, $<t_{0,1}>$ for a given sample x_test
```
combinations = [[0], [0,1]] 
x_test = torch.randn(#batch, #features)
tc_dict = model.get_tc(x_test, combinations)
```
The output is a dict containing the taylorcoefficients $<t_{0}>$, $<t_{0,1}>$.

## Maximal flexibility

This package is designed in a way to allow for maximal flexibility. While the reduction function has to be specified (e.g. mean, median, absolute values etc.) the visualization is up to the user. At this point you should have a look at our [example](example/example.py).


## Authors
- [Lars Sowa](https://github.com/lsowa)
- [Artur Monsch](https://github.com/a-monsch)
