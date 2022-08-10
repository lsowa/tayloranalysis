import os
from setuptools import setup, find_packages

package_dir = os.path.dirname(os.path.abspath(__file__))
requirements_file = os.path.join(package_dir, "requirements.txt")
with open(requirements_file, "r") as rf:
    install_requires = [req.strip() for req in rf.readlines() if req.strip() and not req.startswith("#")]

setup(
    name="Tayloranalysis",
    version="0.0.1",
    author="Lars Sowa, Artur Monsch",
    url="https://github.com/lsowa/tayloranalysis",
    packages=find_packages(),
    description='Taylorcoefficient Analysis for pytorch models',
    long_description='Pytorch implementation of the Paper "Identifying the relevant dependencies of the neural network response on characteristics of the input space" (S. Wunsch, R. Friese, R. Wolf, G. Quast). As in the paper explained, the method computes the averaged taylorcoefficients of a taylored model function. These coefficients are noted as . This is the optimal method to identify not only first order feature importance, but also higher order importance (i.e. the importance of combined features). This module can be applied to each differentiable pytorch model with a scalar output value.',
    install_requires=install_requires,
)
