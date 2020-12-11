from auto_ks.kalman_smoother import kalman_smoother, _kalman_smoother, KalmanSmootherParameters, KalmanSmootherParameterDerivatives
import numpy as np
import time
import copy


def prediction_loss(parameters, y, K, M, lam, grad=True):
    xhat, yhat, DT = kalman_smoother(
        parameters, y, K & ~M, lam)
    L = np.sum(np.square(yhat - y) * M) / M.sum()
    if grad:
        grad_yhat = (2 * (yhat - y) * M).flatten() / M.sum()
        derivatives = DT(dyhat=grad_yhat)
        return L, derivatives, yhat, xhat
    else:
        return L


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
    T, p = y.shape
    n, _ = initial_parameters.A.shape

    parameters = copy.deepcopy(initial_parameters)

    if M is None:
        M = np.zeros(T * p, dtype=int)
        M[:int(fraction * T * p)] = 1
        np.random.shuffle(M)
        M = M.astype(bool).reshape((T, p))
    
    np.testing.assert_array_equal(initial_parameters.A.shape, (n, n))
    np.testing.assert_array_equal(initial_parameters.W_neg_sqrt.shape, (n, n))
    np.testing.assert_array_equal(initial_parameters.C.shape, (p, n))
    np.testing.assert_array_equal(initial_parameters.V_neg_sqrt.shape, (p, p))
    np.testing.assert_array_equal(M.shape, (T, p))
    np.testing.assert_array_equal(K.shape, (T, p))

    assert K.sum() > 0, "Must know at least one measurement."
    assert M.sum() > 0, "Must be at least one missing measurement."

    info = dict()
    info["losses"] = []
    info["parameters"] = []
    info["lrs"] = []

    for k in range(1, niter + 1):
        L, derivatives, yhat, xhat = prediction_loss(
            parameters, y, K, M, lam, grad=True)
        _, r = prox(parameters, lr)
        L += r
        if callback is not None:
            callback(k, yhat)
        info["losses"] += [L]
        info["parameters"] += [copy.deepcopy(parameters)]
        info["lrs"] += [lr]
        if verbose:
            print("%03d | %4.4e | %4.4e" % (k, L, lr))
        while True:
            parameters_next, _ = prox(parameters - lr * derivatives, lr)
            L_next = prediction_loss(
                parameters_next, y, K, M, lam, grad=False)
            _, r = prox(parameters_next, lr)
            L_next += r

            if L_next < L:
                lr *= increase_rate
                break
            elif lr < 1e-10:
                break
            else:
                lr *= decrease_rate
        parameters = parameters_next

    return parameters, info

def tune_forecast(initial_parameters, prox, y, lam, num_splits=20, num_known=20, num_unknown=1,
    niter=200, lr=1.0, increase_rate=1.5, decrease_rate=0.5, verbose=True, callback=None):
    """
    Automatically fit a Kalman Smoother to data.

    Args:
        - initial_parameters: initial KalmanSmootherParameters object
        - prox: Proximal operator for regularization. Returns a
            KalmanSmootherParameters object and value of regularization.
        - y: T x p measurements matrix.
        - lam: regularization parameter.
        - num_splits: number of splits to use
        - num_known: length of known measurements before forecast
        - num_unknown: length of forecast
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
        - prediction_loss_forecast: 
    """
    T, p = y.shape
    n, _ = initial_parameters.A.shape

    parameters = copy.deepcopy(initial_parameters)

    M = np.zeros((num_known + num_unknown, p), dtype=int)
    M[num_known:] = 1
    M = M.astype(bool)
    
    np.testing.assert_array_equal(initial_parameters.A.shape, (n, n))
    np.testing.assert_array_equal(initial_parameters.W_neg_sqrt.shape, (n, n))
    np.testing.assert_array_equal(initial_parameters.C.shape, (p, n))
    np.testing.assert_array_equal(initial_parameters.V_neg_sqrt.shape, (p, p))

    info = dict()
    info["losses"] = []
    info["parameters"] = []
    info["lrs"] = []

    def prediction_loss_forecast(parameters, y, grad=False):
        step = (T - num_known - num_unknown) // num_splits
        smooth, DT = _kalman_smoother(
            parameters, ~M, lam)
        L = 0.
        derivatives = copy.deepcopy(parameters) * 0.
        for i in range(num_splits):
            yi = y[i * step: i * step + num_known + num_unknown]
            xhat, yhat, zhat = smooth(yi)
            L += np.sum(np.square(yhat - yi) * M) / M.sum()
            if grad:
                grad_yhat = (2 * (yhat - yi) * M).flatten() / M.sum()
                derivatives += DT(zhat, dyhat=grad_yhat)
        if grad:
            return L, derivatives, yhat, xhat
        else:
            return L

    for k in range(1, niter + 1):
        L, derivatives, yhat, xhat = prediction_loss_forecast(parameters, y, grad=True)
        _, r = prox(parameters, lr)
        L += r
        if callback is not None:
            callback(k, yhat, parameters, prediction_loss_forecast)
        info["losses"] += [L]
        info["parameters"] += [copy.deepcopy(parameters)]
        info["lrs"] += [lr]
        if verbose:
            print("%03d | %4.4e | %4.4e" % (k, L, lr))
        while True:
            parameters_next, _ = prox(parameters - lr * derivatives, lr)
            L_next = prediction_loss_forecast(parameters_next, y)
            _, r = prox(parameters_next, lr)
            L_next += r

            if L_next < L:
                lr *= increase_rate
                break
            elif lr < 1e-10:
                break
            else:
                lr *= decrease_rate
        parameters = parameters_next

    return parameters, info, prediction_loss_forecast
