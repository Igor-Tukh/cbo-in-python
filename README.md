# Consensus-Based Optimization in Python

## Description
`CBO-in-python` is a library for working with Consensus-Based Optimization (**CBO**) in Python. The library provides an interface to perform CBO to minimize functions and train neural networks. 

To conveniently work with neural networks, we deliver our library as two models: one for working with `PyTorch` and one for working with `TensorFlow` (two major machine learning frameworks). Nevertheless, our general focus is on the PyTorch module. While we provide limited TensorFlow support and some usage examples, we recommend using the PyTorch module. We are no longer updating the TensorFlow module, and the current implementation might include bugs.


## Setup
### virtualenv
One may use [virtualenv](https://pypi.org/project/virtualenv/) to prepare an isolated environment for the library:
* Install virtualenv:
```
pip install virtualenv
```
* Create a new environment:
```
virtualenv -p python3 ~/virtualenvs/cbo
```
* Activate new environment:
```
source ~/virutalenvs/cbo/bin/activate
```
* Install the dependencies:
```
pip install -r requirements.txt
```

Alternatively, one can install the missing libraries via the following command:
```
pip install -r requirements.txt
```
It is highly recommended to use environments when working with Python dependencies, though. One may consider using `conda` for enviroments and packages managing as an alternative approach.

## Examples
### Demo scripts
We provide a few demo CLI scripts in [demo folder](https://github.com/Igor-Tukh/cbo-in-python/tree/master/demo).

### Notebooks
* Please refer to [`notebooks/torch_cbo_on_mnist.ipynb`](https://github.com/Igor-Tukh/cbo-in-python/blob/master/notebooks/torch_cbo_on_mnist.ipynb) for details on torch implementation of CBO.
* Please refer to [`notebooks/minimize_pytorch.ipynb`](https://github.com/Igor-Tukh/cbo-in-python/blob/master/notebooks/minimize_pytorch.ipynb) for  examples on functions minimization and visualization via `PyTorch` implementation.
* Please refer to [`notebooks/functions_minimization_examples.ipynb`](https://github.com/Igor-Tukh/cbo-in-python/blob/master/notebooks/functions_minimization_examples.ipynb) to view examples on functions minimization and visualization via Tensorflow implementation.
* Some experiments on training neural networks via CBO can be found in folder [`notebooks/experiments`](https://github.com/Igor-Tukh/cbo-in-python/tree/master/notebooks/experiments).
