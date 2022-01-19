# Taylorcoefficient Analysis
Pytorch implementation of the Paper 
["Identifying the relevant dependencies of the neural network response on characteristics of the input space"](https://arxiv.org/abs/1803.08782) 
(S. Wunsch, R. Friese, R. Wolf, G. Quast)

The module computes the taylorceofficients $`t_{i} :=\partial_{i} \text{NN}(\vec{a})`$ of the taylored NN($`\vec{x}`$) function
```math
\text{NN}(\vec{x}) \approx & \text{ NN}(\vec{a}) \\ 
                        & +  \partial_{x_1} \text{NN}(\vec{a}) + \partial_{x_2} \text{NN}(\vec{a})\\
                        & + \frac{1}{2} \partial_{x_1x_1} \text{NN}(\vec{a}) + \frac{1}{2} \partial_{x_1x_2} \text{NN}(\vec{a}) + \frac{1}{2} \partial_{x_2x_2} \text{NN}(\vec{a}) \\
                        & + \text{ ... } 
```
and averages over the inpit batch
```math
\left< t_i \right> = \frac{1}{N} \sum^N_k \left| t_i(\vec{a}_k) \right| && \text{for example $k$ in dataset N}.
```





