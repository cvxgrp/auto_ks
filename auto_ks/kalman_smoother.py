import numpy as np
import numpy.random as npr
from scipy import sparse
from scipy.sparse import linalg as splinalg
import IPython as ipy
import time
import numbers

class KalmanSmootherParameters:
    def __init__(self, A, W_neg_sqrt, C, V_neg_sqrt):
        self.A = A
        self.W_neg_sqrt = W_neg_sqrt
        self.C = C
        self.V_neg_sqrt = V_neg_sqrt

    def __add__(self, y):
        if isinstance(y, KalmanSmootherParameters):
            return KalmanSmootherParameters(self.A + y.A, self.W_neg_sqrt + y.W_neg_sqrt, self.C + y.C, self.V_neg_sqrt + y.V_neg_sqrt) 
        elif isinstance(y, KalmanSmootherParameterDerivatives):
            return KalmanSmootherParameters(self.A + y.DA, self.W_neg_sqrt + y.DW_neg_sqrt, self.C + y.DC, self.V_neg_sqrt + y.DV_neg_sqrt) 
        else:
            return NotImplementedError

    def __sub__(self, y):
        return self.__add__(-1.0 * y)

    def __mul__(self, a):
        assert isinstance(a, numbers.Number)
        return KalmanSmootherParameters(self.A * a, self.W_neg_sqrt * a, self.C * a, self.V_neg_sqrt * a) 
    
    __radd__ = __add__
    __rmul__ = __mul__

class KalmanSmootherParameterDerivatives:
    def __init__(self, DA, DW_neg_sqrt, DC, DV_neg_sqrt):
        self.DA = DA
        self.DW_neg_sqrt = DW_neg_sqrt
        self.DC = DC
        self.DV_neg_sqrt = DV_neg_sqrt

    def __mul__(self, a):
        assert isinstance(a, numbers.Number)
        return KalmanSmootherParameters(self.DA * a, self.DW_neg_sqrt * a, self.DC * a, self.DV_neg_sqrt * a)
    
    __rmul__ = __mul__

def get_D(A, W_neg_sqrt, C, V_neg_sqrt, n, p, T, lam):
    temp1 = sparse.kron(sparse.eye(T - 1),
                        W_neg_sqrt)
    temp2 = sparse.kron(sparse.eye(T - 1), -W_neg_sqrt @ A)
    D_11 = sparse.hstack([sparse.csc_matrix(((T - 1) * n, n)), temp1]) + sparse.hstack([
        temp2, sparse.csc_matrix(((T - 1) * n, n))])
    D_12 = sparse.csc_matrix(((T - 1) * n, T * p))
    D_21 = sparse.kron(sparse.eye(T), -V_neg_sqrt @ C)
    D_22 = sparse.kron(sparse.eye(T), V_neg_sqrt)

    return sparse.bmat([
        [D_11, D_12],
        [D_21, D_22],
        [lam * sparse.eye(T * n), None],
        [None, lam * sparse.eye(T * p)]
    ])

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
    T, p = y.shape
    assert y.ndim == 2
    assert type(y) is np.ndarray
    np.testing.assert_array_equal(y.shape, (T, p))

    solve, DT = _kalman_smoother(kalman_smoother_parameters, K, lam)
    xhat, yhat, z = solve(y)
    def DT1(dxhat=np.zeros(xhat.shape), dyhat=np.zeros(yhat.shape)):
        return DT(z, dxhat=dxhat, dyhat=dyhat)
    return xhat, yhat, DT1 

def _kalman_smoother(kalman_smoother_parameters, K, lam):
    """
    minimize    ||Dz||^2
    subject to  Bz=c

    Args:
        - kalman_smoother_paramters: KalmanSmootherParameters object.
        - K: T x p boolean output mask
        - lam: float, scale of Tikhonov regularization

    Returns:
        - solve: a method that takes one argument: y, and smooths it
        - DT: function that computes derivative
    """
    A = kalman_smoother_parameters.A
    W_neg_sqrt = kalman_smoother_parameters.W_neg_sqrt
    C = kalman_smoother_parameters.C
    V_neg_sqrt = kalman_smoother_parameters.V_neg_sqrt

    T, _ = K.shape
    n, _ = A.shape
    p = V_neg_sqrt.shape[0]
    z_size = (n + p) * T

    # First we form the least squares coefficient matrix D
    D = get_D(A, W_neg_sqrt, C, V_neg_sqrt, n, p, T, lam)
    D_full = get_D(np.ones(A.shape), np.ones(
        W_neg_sqrt.shape), np.ones(C.shape), np.ones(V_neg_sqrt.shape), n, p, T, 1)
    D_rows, D_cols = D_full.nonzero()
    del D_full

    assert type(A) is np.ndarray or type(A) is np.matrix
    assert type(W_neg_sqrt) is np.ndarray or type(W_neg_sqrt) is np.matrix
    assert type(C) is np.ndarray or type(C) is np.matrix
    assert type(V_neg_sqrt) is np.ndarray or type(V_neg_sqrt) is np.matrix
    assert type(K) is np.ndarray
    assert isinstance(lam, numbers.Number) 
    assert A.ndim == 2


    np.testing.assert_array_equal(A.shape, (n, n))
    np.testing.assert_array_equal(A.shape, (n, n))
    np.testing.assert_array_equal(W_neg_sqrt.shape, (n, n))
    np.testing.assert_array_equal(C.shape, (p, n))
    np.testing.assert_array_equal(V_neg_sqrt.shape, (p, p))
    np.testing.assert_array_equal(K.shape, (T, p))

    # Next we form the coefficients of the equality constraint
    rows, cols = K.nonzero()
    c_size = K.sum()
    S = sparse.csc_matrix((np.ones(c_size), (np.arange(
        c_size), rows * p + cols)), shape=(c_size, T * p))
    B = sparse.bmat([
        [sparse.csc_matrix((c_size, T * n)), S]
    ])

    # Next we form the KKT matrix
    M = sparse.bmat([
        [None, D.T, B.T],
        [D, -sparse.eye(D.shape[0]), None],
        [B, None, None]
    ], format='csc')

    # And factorize it
    solve = splinalg.factorized(M)

    def smooth(y):
        c = y[K]
        # And solve for z
        rhs = np.concatenate([np.zeros(z_size), np.zeros(D.shape[0]), c])
        sol = solve(rhs)
        z = sol[:z_size]

        xhat = z[:T * n].reshape(T, n, order='C')
        yhat = z[T * n:T * (n + p)].reshape(T, p, order='C')

        return xhat, yhat, z

    # This function implements the derivative
    def DT(z, dxhat=np.zeros((T, n)), dyhat=np.zeros((T, p))):
        """
        Args:
            - dxhat: T x n output trajectory
            - dyhat: T x p output trajectory
        """
        g = np.concatenate(
            [dxhat.flatten(order='C'), dyhat.flatten(order='C'), np.zeros(D.shape[0]), np.zeros(c_size)])
        dsol = -solve(g)[:z_size]

        values = (D @ z)[D_rows] * dsol[D_cols] + (D @ dsol)[D_rows] * z[D_cols]
        dD = sparse.csc_matrix((values, (D_rows, D_cols)), shape=D.shape)

        DA = np.zeros(A.shape)
        DW_neg_sqrt = np.zeros(W_neg_sqrt.shape)
        DC = np.zeros(C.shape)
        DV_neg_sqrt = np.zeros(V_neg_sqrt.shape)

        summer_T_left = sparse.kron(np.ones((1, T)), sparse.eye(p))
        summer_T_right = sparse.kron(np.ones((T, 1)), sparse.eye(p))
        summer_T_right_n = sparse.kron(np.ones((T, 1)), sparse.eye(n))
        summer_T1_left = sparse.kron(np.ones((1, T - 1)), sparse.eye(n))
        summer_T1_right = sparse.kron(np.ones((T - 1, 1)), sparse.eye(n))

        inside = -summer_T_left @ dD[(T - 1) * n:(T - 1) * n + T * p, :T * n] @ summer_T_right_n
        DV_neg_sqrt += summer_T_left @ dD[(T - 1) * n:(T - 1) * n + T * p, T * n:T * n + T * p] @ summer_T_right
        DV_neg_sqrt += inside @ C.T
        DC += V_neg_sqrt.T @ inside

        masked = dD[:(T - 1) * n, :(T - 1) * n]
        masked = masked.multiply(sparse.kron(
            sparse.eye(T - 1), np.ones((n, n))))
        inside = -summer_T1_left @ masked @ summer_T1_right
        DA += W_neg_sqrt.T @ inside
        DW_neg_sqrt += inside @ A.T
        masked = dD[:(T - 1) * n, n:T * n]
        masked = masked.multiply(sparse.kron(
            sparse.eye(T - 1), np.ones((n, n))))
        DW_neg_sqrt += summer_T1_left @ masked @ summer_T1_right

        DA = np.array(DA)
        DW_neg_sqrt = np.array(DW_neg_sqrt)
        DC = np.array(DC)
        DV_neg_sqrt = np.array(DV_neg_sqrt)

        return KalmanSmootherParameterDerivatives(DA, DW_neg_sqrt, DC, DV_neg_sqrt)

    return smooth, DT
