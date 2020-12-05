import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import time

from auto_ks import kalman_smoother, KalmanSmootherParameters


np.random.seed(1)


def profile_random_instance(n, p, T):
    lam = 1e-6
    A = npr.randn(n, n)
    W_neg_sqrt = npr.randn(n, n)
    C = npr.randn(p, n)
    V_neg_sqrt = npr.randn(p, p)

    y = npr.randn(T, p)
    K = np.zeros(T * p, dtype=int)
    K[:int(.5 * T * p)] = 1
    np.random.shuffle(K)
    K = K.astype(bool).reshape((T, p))

    tic = time.time()
    params = KalmanSmootherParameters(A, W_neg_sqrt, C, V_neg_sqrt)
    xhat, yhat, DT = kalman_smoother(params, y, K, lam)
    toc = time.time()
    forward_time = toc - tic

    tic = time.time()
    DT(y)
    toc = time.time()
    backward_time = toc - tic

    return forward_time * 1000, backward_time * 1000


def prof(n, p, T, num=10):
    x = [profile_random_instance(n, p, T) for _ in range(num)]
    forwards = [t[0] for t in x]
    backwards = [t[1] for t in x]
    return np.mean(forwards), np.std(forwards), np.mean(backwards), np.std(backwards)

if __name__ == '__main__':

    for T in [10, 100, 1_000, 10_000]:
        fwd_mean, fwd_std, bkwd_mean, bkwd_std = prof(10, 10, T)

        print("%d & %.1f $\\pm$ %.1f & %.1f $\\pm$ %.1f \\\\" %
              (T, fwd_mean, fwd_std, bkwd_mean, bkwd_std))
