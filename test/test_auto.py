import unittest
import auto_ks
import numpy as np
import numpy.random as npr
from scipy.linalg import sqrtm


class TestAuto(unittest.TestCase):

    def test_smoke_test(self):
        np.random.seed(0)
        n = 2
        p = 1
        T = 1000
        lam = 1e-5

        A_true = np.array([
            [1., 1.],
            [0, 1.],
        ])
        C_true = np.array([
            [1., 1.]
        ])
        W_true = .01 * np.eye(n)
        V_true = .1 * np.eye(p)

        # Get trajectory
        x = [npr.randn(n)]
        y = [C_true @ x[0] + np.random.multivariate_normal(np.zeros(p), V_true)]
        for t in range(T - 1):
            xt = x[-1]
            x.append(A_true @ xt + np.random.multivariate_normal(np.zeros(n), W_true))
            y.append(C_true @ xt + np.random.multivariate_normal(np.zeros(p), V_true))
        x = np.array(x)
        y = np.array(y)

        K = np.ones((T, p), dtype=np.bool)

        W_neg_sqrt_true = sqrtm(np.linalg.inv(W_true))
        V_neg_sqrt_true = sqrtm(np.linalg.inv(V_true))

        def prox(params, t):
            return auto_ks.KalmanSmootherParameters(params.A, params.W_neg_sqrt, params.C, params.V_neg_sqrt), 0.0

        params = auto_ks.KalmanSmootherParameters(
            A_true + 1e-2*np.random.randn(n, n),
            W_neg_sqrt_true,
            C_true,
            V_neg_sqrt_true + .1*np.eye(p)
        )

        params, info = auto_ks.tune(params, prox, y, K, lam, verbose=True, niter=25, lr=1e-3)

        np.testing.assert_array_less(np.diff(info["losses"]), 0.0)

if __name__ == '__main__':
    unittest.main()
