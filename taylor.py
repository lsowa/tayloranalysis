from pickle import TRUE
import matplotlib.pyplot as plt
import numpy as np
import torch

from datetime import datetime
from torch import nn
from torch.autograd.functional import jacobian
from torch.utils.data import TensorDataset, DataLoader


def seq_exec(func):
    """Wrap a function with input data, execute the function sequentially on the 
    data and merge the results. This is meant to avoid memory allocation errors."""
    def wrapped_func(self, data, order):
        batch_size=data.shape[0]
        while True:
            try:
                output = []
                loader = DataLoader(TensorDataset(data), batch_size=batch_size)
                for dat in loader:
                    out = func(self, dat[0], order) # output is a np.array
                    output.append(out)
                return np.stack(output).mean(axis=0)
            except RuntimeError: # If cuda is out of memory -> reduce batch size
                batch_size = int(batch_size / 2)
                print('\tReduce sequential batch size to', batch_size, ' at ', datetime.now().time())
                pass
    return wrapped_func


class TaylorAnalysis(nn.Module):
    """Wrapper to add taylor coefficient analysis to a differentiable pytorch model.

    Args:
        nn (nn.Module): nn.Module
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.checkpoints_first_order = []
        self.checkpoints_second_order = []
        self.checkpoints_third_order = []

    def forward(self, x):
        return self.model(x)

    @seq_exec
    def tc_mean(self, data, order=2):
        '''Calc <t_i> and return it as num,py array (!)'''
        self.order = order
        derivated_model_func = self._choose_order(order)
        data = torch.abs(derivated_model_func(data))
        data = torch.mean(data, dim=0)
        return data.cpu().detach().numpy()

    #@seq_exec(100)
    def _first_order(self, x_data):
        '''first order derivative'''
        return jacobian(self.__call__, x_data, create_graph=self.order>1, strict=True, vectorize=False).sum(-2)

    #@seq_exec(40)
    def _second_order(self, x_data):
        '''second order derivative'''
        return jacobian(self._first_order, x_data, create_graph=self.order>2, strict=True, vectorize=False).sum(-2)

    #@seq_exec(15)
    def _third_order(self, x_data):
        '''third order derivative'''
        return jacobian(self._second_order, x_data, create_graph=self.order>3, strict=True, vectorize=False).sum(-2)
    
    def _choose_order(self, order):
        if order == 1: return self._first_order
        elif order == 2: return self._second_order
        elif order == 3: return self._third_order

    def plot_tc(self, data, names, path, order=2):
        '''Plot <t_i> for given input data.'''

        # first order
        print('Plotting first order coefficients...')
        derivatives = self.tc_mean(data, order=1)
        for i in range(len(names)):
            plt.plot('$t_{{{}}}$'.format(names[i]), derivatives[i], 
                    '+', color='black', markersize=10)

        # second order
        print('Plotting second order coefficients...')
        if order >=2:
            derivatives = self.tc_mean(data, order=2)
            for i in range(len(names)):
                for j in range(len(names)):
                    if i<=j: # ignore diagonal elements
                        plt.plot('$t_{{{},{}}}$'.format(names[i], names[j]), 
                                derivatives[i,j], '+', color='black', markersize=10)

        # third order
        print('Plotting third order coefficients...')
        if order >=3:
            derivatives = self.tc_mean(data, order=3)
            for i in range(len(names)):
                for j in range(len(names)):
                    for k in range(len(names)):
                        if i<=j<=k:  # ignore diagonal elements
                            plt.plot('$t_{{{},{},{}}}$'.format(names[i], names[j], names[k]), 
                                    derivatives[i,j,k], '+', color='black', markersize=10)

        plt.ylabel('$<t_i>$', loc='top', fontsize=13)
        plt.xticks(rotation=45)
        plt.tick_params(axis='y', which='both', right=True, direction='in')
        plt.tick_params(axis='x', which='both', top=True, direction='in')
        plt.savefig(path+'coefficients.pdf', bbox_inches = "tight")
        plt.clf()

    def tc_checkpoint(self, x_data, order=1):
        '''Save coefficients during training.'''
        self.eval()
        if order >= 1:
            self.checkpoints_first_order.append(self.tc_mean(x_data, order=1))
        if order >= 2:
            self.checkpoints_second_order.append(self.tc_mean(x_data, order=2))
        if order >= 3:
            self.checkpoints_third_order.append(self.tc_mean(x_data, order=3))
        self.train()
        
    def tc_plt_checkpoints(self, names, path):
        '''Plot saved coefficients from training.'''

        # first order
        if self.checkpoints_first_order:
            derivatives_first = np.stack(self.checkpoints_first_order)
            for i, name_i in enumerate(names):
                plt.plot(derivatives_first[:,i], label='$t_{{{}}}$'.format(name_i))

                # second order
                if self.checkpoints_second_order:
                    derivatives_second = np.stack(self.checkpoints_second_order)
                    for j, name_j in enumerate(names):
                        if i<=j: # ignore diagonal elements
                            plt.plot(derivatives_second[:,i,j], 
                                    label='$t_{{{},{}}}$'.format(name_i, name_j))

                            # third order
                            if self.checkpoints_third_order:
                                derivatives_third = np.stack(self.checkpoints_third_order)
                                for k, name_k in enumerate(names):
                                    if i<=j<=k: # ignore diagonal elements
                                        plt.plot(derivatives_third[:,i,j,k], 
                                                label='$t_{{{},{},{}}}$'.format(name_i, 
                                                                                name_j, 
                                                                                name_k))

        plt.legend()
        plt.xlabel('Epoch', loc='right', fontsize=13)
        plt.ylabel('$<t_i>$', loc='top', fontsize=13)
        plt.tick_params(axis='y', which='both', right=True, direction='in')
        plt.tick_params(axis='x', which='both', top=True, direction='in')
        plt.savefig(path+'tc_training.pdf', bbox_inches = "tight")
        plt.clf()




    











