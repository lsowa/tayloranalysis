#!/usr/bin/env python

from setuptools import setup, find_packages

long_description='''Pytorch implementation of the Paper "Identifying the relevant dependencies of the neural network response on characteristics of the input space" (S. Wunsch, R. Friese, R. Wolf, G. Quast). As in the paper explained, the method computes the averaged taylorcoefficients of a taylored model function. These coefficients are noted as . This is the optimal method to identify not only first order feature importance, but also higher order importance (i.e. the importance of combined features). This module can be applied to each differentiable pytorch model with a scalar output value.
'''

setup(name='tayloranalysis',
      version='0.0.1',
      description='Taylorcoefficient Analysis for pytorch models',
      long_description=long_description,
      author='Lars Sowa',
      author_email='lars.sowa@kit.edu',
      url='https://github.com/lsowa/tayloranalysis',
      download_url='https://github.com/lsowa/tayloranalysis',
      packages=find_packages()
     )
