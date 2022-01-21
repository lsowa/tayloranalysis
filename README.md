# Taylorcoefficient Analysis
Pytorch implementation of the Paper 
["Identifying the relevant dependencies of the neural network response on characteristics of the input space"](https://arxiv.org/abs/1803.08782) 
(S. Wunsch, R. Friese, R. Wolf, G. Quast)

As in the paper explained, the method computes the averaged taylorcoefficients of a taylored model function. These coefficients are noted as <img src="https://render.githubusercontent.com/render/math?math=<t_i>">. 

This is the optimal method to identify not only first order feature importance, but also higher order importance (i.e. the importance of combined features).

This module can be applied to each differentiable pytorch model with a scalar output value.

## Installation
```
pip install git+https://github.com/lsowa/tayloranalysis.git
```

## Usage

Setup your data and model, all you have to do is to wrap your model with the `TaylorAnalysis` class. 
```
import tayloranalysis
...
model = Mlp()
model = TaylorAnalysis(model)
...
for epoch in epochs:
    ...
    # save taylorcoefficients during training
    model.tc_checkpoint(x_train, names=['x1', 'x2'], order=3)
    
# plot taylorcoefficients after training
model.plot_tc(data=x_test, names=['x1', 'x2'], path='', order=3)

# plot saved checkpoints
model.plt_checkpoints(path='')
```
Note that your data should be of shape (batch, features). `names` should be a list of all features in the same order as in the feature dimension of your data.

## Resluts

![Plottet Taylorcoefficients after Training](/lsowa/tayloranalysis/blob/master/example/coefficients.png)

![Plotted Checkpoints](https://raw.githubusercontent.com/lsowa/tayloranalysis/blob/master/example/tc_training.png)

