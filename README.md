# auto_ks

Implementation of the paper ["Fitting a Kalman Smoother to Data"](http://web.stanford.edu/~boyd/papers/auto_ks.html).

## Installation

To install, run:
```bash
pip install git+https://github.com/cvxgrp/auto_ks.git
```

Install extra packages to run the examples:
```bash
pip install -r requirements.txt
```

## Usage

To smooth a given dataset:
```python
def kalman_smoother(kalman_smoother_parameters, y, K, lam):
    """
    minimize    ||Dz||^2
    subject to  Bz=c

    Args:
        - kalman_smoother_paramters: KalmanSmootherParameters object.
        - y: T x p output trajectory
        - K: T x p boolean output mask
        - lam: float, scale of Tikhonov regularization

    Returns:
        - xhat: state trajectory
        - yhat: output trajectory
        - DT: function that computes derivative
    """
```

To fit the parameters to a dataset:
```python
def tune(initial_parameters, prox, y, K, lam, M=None, niter=200, lr=1.0, fraction=0.5,
         increase_rate=1.5, decrease_rate=0.5, verbose=True, callback=None):
    """
    Automatically fit a Kalman Smoother to data.

    Args:
        - initial_parameters: initial KalmanSmootherParameters object
        - prox: Proximal operator for regularization. Returns a
            KalmanSmootherParameters object and value of regularization.
        - y: T x p measurements matrix.
        - K: T x p mask matrix of known measurements.
        - lam: regularization parameter.
        - M (optional): T x p mask matrix of missing measurements. Defaults to
            dropping "fraction" of measurements. (Default=None)
        - niter (optional): Number of iterations. (Default=200)
        - lr (optional): Initial learning rate. (Default=1.0)
        - fraction (optional): Fraction of measurements to drop. (Default=0.5)
        - increase_rate (optional): Rate to increase learning rate. (Default=1.5)
        - decrease_rate (optional): Rate to decrease learning rate. (Default=0.5)
        - verbose (optional): Whether or not to print iterations. (Default=True)
        - callback (optional): Callback function to be called every iteration. (Default=None)
    Returns:
        - parameters: KalmanSmootherParameters result.
        - info: dictionary of results.
    """
```

## Run tests

To run tests:
```bash
cd test
python -m unittest
```

## Run examples
To run examples:
```bash
cd examples
python human_migration.py
```

## Citing
If you use `auto_ks` in your research, please consider citing our paper:
```
@article{barratt2019fitting,
  title={Fitting a Kalman Smoother to Data},
  author={Barratt, Shane and Boyd, Stephen},
  journal={arXiv preprint arXiv:1910.08615},
  year={2019}
}
```
