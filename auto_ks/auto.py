from auto_ks.kalman_smoother import kalman_smoother, KalmanSmootherParameters, KalmanSmootherParameterDerivatives
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
    T, p = y.shape
    n, _ = initial_parameters.A.shape

    parameters = copy.deepcopy(initial_parameters)

    M = np.zeros(T * p, dtype=int)
    M[:int(fraction * T * p)] = 1
    np.random.shuffle(M)
    M = M.astype(bool).reshape((T, p))

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
