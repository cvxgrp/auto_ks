import unittest
import auto_ks
import numpy as np
from numpy import random as npr
import cvxpy as cp


class TestKalmanFilter(unittest.TestCase):

    def test_kalman_filter_vs_cvxpy(self):
        np.random.seed(0)

        n = 3
        p = 6
        T = 10
        lam = 1e-10

        for _ in range(5):

            A = npr.randn(n, n)
            W_neg_sqrt = npr.randn(n, n)
            C = npr.randn(p, n)
            V_neg_sqrt = npr.randn(p, p)

            y = npr.randn(T, p)
            K = np.zeros(T * p, dtype=int)
            K[:int(.7 * T * p)] = 1
            np.random.shuffle(K)
            K = K.astype(bool).reshape((T, p))

            params = auto_ks.KalmanSmootherParameters(A, W_neg_sqrt, C, V_neg_sqrt)
            xhat, yhat, DT = auto_ks.kalman_smoother(
                params, y, K, lam)

            xs = cp.Variable((T, n))
            ys = cp.Variable((T, p))
            objective = 0.0
            for t in range(T):
                objective += cp.sum_squares(V_neg_sqrt @ (ys[t] - C @ xs[t]))
                if t < T - 1:
                    objective += cp.sum_squares(W_neg_sqrt @ (xs[t + 1] - A @ xs[t]))
            objective += cp.sum_squares(lam * xs) + cp.sum_squares(lam * ys)
            constraints = []
            rows, cols = K.nonzero()
            for t, i in zip(rows, cols):
                constraints += [ys[t, i] == y[t, i]]
            prob = cp.Problem(cp.Minimize(objective), constraints)
            prob.solve(solver='OSQP')
            np.testing.assert_allclose(xhat, xs.value)
            np.testing.assert_allclose(yhat, ys.value)

    def test_kalman_filter_derivative(self):
        np.random.seed(0)

        n = 5
        p = 10
        T = 100
        lam = 1e-10

        for _ in range(10):
            A = npr.randn(n, n)
            W_neg_sqrt = np.eye(n)
            C = npr.randn(p, n)
            V_neg_sqrt = npr.randn(p, p)

            y = npr.randn(T, p)
            K = np.zeros(T * p, dtype=int)
            K[:int(.7 * T * p)] = 1
            np.random.shuffle(K)
            K = K.astype(bool).reshape((T, p))

            params = auto_ks.KalmanSmootherParameters(A, W_neg_sqrt, C, V_neg_sqrt)
            xhat, yhat, DT = auto_ks.kalman_smoother(
                params, y, K, lam)
            f = np.sum(xhat) + np.sum(yhat)

            dxhat = np.ones(xhat.shape)
            dyhat = np.ones(yhat.shape)

            derivatives = DT(dxhat, dyhat)
            eps = 1e-6
            xhat_p, yhat_p, _ = auto_ks.kalman_smoother(params + eps * derivatives, y, K, lam)
            fp = np.sum(xhat_p) + np.sum(yhat_p)
            increase = fp - f
            dA = derivatives.DA
            dW_neg_sqrt = derivatives.DW_neg_sqrt
            dC = derivatives.DC
            dV_neg_sqrt = derivatives.DV_neg_sqrt
            predicted_increase = eps * (np.sum(dA * dA) + np.sum(
                dW_neg_sqrt * dW_neg_sqrt) + np.sum(dC * dC) + np.sum(dV_neg_sqrt * dV_neg_sqrt))
            np.testing.assert_allclose(predicted_increase, increase, atol=1e-5)

if __name__ == '__main__':
    unittest.main()
