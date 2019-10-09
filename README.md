# auto_ks

Implementation of "Fitting a Kalman Smoother to Data".

## Installation

To install:
```bash
python setup.py install
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
def tune(initial_parameters, prox, y, K, M, lam, niter=200, lr=1.0, fraction=0.5,
         increase_rate=1.5, decrease_rate=0.5, verbose=True, callback=None):
    """
    Automatically fit a Kalman Smoother to data.

    Args:
        - initial_parameters: initial KalmanSmootherParameters object
        - prox: Proximal operator for regularization. Returns a KalmanSmootherParameters object and value of regularization.
        - y: T x p measurements matrix.
        - K: T x p mask matrix of known measurements.
        - M: T x p mask matrix of missing measurements.
        - lam: regularization parameter.
        - niter: Number of iterations.
        - lr: Initial learning rate.
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