import numpy as np
import auto_ks

T = 100
n = 5
m = 3
A = np.random.randn(n, n)
A /= (1.1 * np.abs(np.linalg.eig(A)[0]).max())
C = np.random.randn(m, n)
V = .1 * np.eye(m)
W = .1 * np.eye(n)

x = np.zeros(n)
ys = []
for _ in range(T):
	x = np.random.multivariate_normal(A @ x, W)
	y = np.random.multivariate_normal(C @ x, V)
	ys.append(y)

ys = np.array(ys)

x = np.zeros(n)
ys_test = []
for _ in range(T):
	x = np.random.multivariate_normal(A @ x, W)
	y = np.random.multivariate_normal(C @ x, V)
	ys_test.append(y)

ys_test = np.array(ys_test)

parameters = auto_ks.KalmanSmootherParameters(np.random.randn(n, n),
	np.eye(n) * .5,
	np.random.randn(m, n),
	np.eye(m) * .5)

def prox(params, t):
	return params, 0.

def callback(k, yhat, parameters, prediction_loss_forecast):
	print("TEST", prediction_loss_forecast(parameters, ys_test))

parameters, info, prediction_loss_forecast = auto_ks.tune_forecast(parameters, prox, ys, 1e-4, num_splits=20,
	num_known=20, num_unknown=1, niter=100, callback=callback)