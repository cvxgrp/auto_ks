import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import g
from scipy.spatial.transform import Rotation as R
import navpy
import auto_ks
from pykalman import KalmanFilter
import IPython as ipy
from random import shuffle
from itertools import cycle
from utils import latexify
import time

np.random.seed(0)

# gather data
# units are in 1m people
df = pd.read_csv("data/state_pop_Annual.txt",
                 delimiter="\t", parse_dates=["DATE"])
df.drop(["AKPOP", "HIPOP"], axis=1, inplace=True)
columns = df.columns[1:]
y = np.array(df.drop(["DATE"], axis=1)) / 1000.0

# construct initial parameters
T, n = y.shape
m = n
lam = 1e-10
A = np.eye(n)
W_neg_sqrt = np.sqrt(1. / .001) * np.diag(np.random.uniform(0.8, 1.2, size=n))
C = np.eye(m)
V_neg_sqrt = np.sqrt(1. / .001) * np.diag(np.random.uniform(0.8, 1.2, size=n))
params = auto_ks.KalmanSmootherParameters(A, W_neg_sqrt, C, V_neg_sqrt)

K = np.zeros(y.shape, dtype=bool)
M = np.zeros(y.shape, dtype=bool)
M_test = np.zeros(y.shape, dtype=bool)
for t in range(T):
    idx = np.arange(n)
    np.random.shuffle(idx)
    indices = idx[:30]
    K[t, indices] = True
    np.random.shuffle(indices)
    M[t, indices[:12]] = True
    M_test[t, indices[12:12 + 5]] = True


def prox(params, t):
    return auto_ks.KalmanSmootherParameters(
        np.maximum(params.A, 0),
        W_neg_sqrt,
        C,
        np.diag(np.maximum(np.diag(params.V_neg_sqrt), 0))
    ), 0.0


def test_loss(params):
    xhat, yhat, _ = auto_ks.kalman_smoother(params, y, K & ~M_test, lam)
    test_loss = np.linalg.norm(yhat[M_test] - y[M_test])**2 / M_test.sum()
    return test_loss

print("starting test loss:", test_loss(params))
tic = time.time()
params, info = auto_ks.tune(
    params, prox, y, K & ~M_test, lam, M=M, lr=1e-4, verbose=True, niter=50)
toc = time.time()
print("time", toc - tic, "s")
xhat, yhat, _ = auto_ks.kalman_smoother(params, y, K & ~M_test, lam)
print("ending test_loss:", test_loss(params))

measurements = np.ma.masked_array(y)
measurements[~(K & ~M_test)] = np.ma.masked
kf = KalmanFilter(transition_matrices=A, observation_matrices=C, transition_covariance=np.linalg.inv(W_neg_sqrt @ W_neg_sqrt), observation_covariance=np.linalg.inv(V_neg_sqrt @ V_neg_sqrt))
kf = kf.em(measurements, n_iter=10)
filtered_state_means, _ = kf.smooth(measurements)
yhat_em = filtered_state_means @ C.T
loss_em = np.linalg.norm(yhat_em[M_test] - y[M_test])**2 / M_test.sum()
print("loss_em:", loss_em)

sorted_off_diagonal = np.sort(
    (params.A - np.diag(np.diag(params.A))).flatten())

for k in range(1, 6):
    i, j = np.where(params.A == sorted_off_diagonal[-k])
    print("From", columns[j], "to", columns[i], sorted_off_diagonal[-k])

V = np.linalg.inv(params.V_neg_sqrt * params.V_neg_sqrt)
i, j = np.where(V == np.max(V))
print("most noisy:", "From", columns[j],
      "to", columns[i], np.sqrt(V[i, j]) * 1e6)

V[V == 0] = 100000
i, j = np.where(V == np.min(V))
print("least noisy", "From", columns[j],
      "to", columns[i], np.sqrt(V[i, j]) * 1e6)

latexify(4, 3)

plt.plot(range(1900, 2019), yhat[:, 3], '-', c='black', label="CA")
plt.scatter(range(1900, 2019), y[:, 3], s=1, alpha=.5, c='black')

plt.plot(range(1900, 2019), yhat[:, 40], '--', c='black', label="TX")
plt.scatter(range(1900, 2019), y[:, 40], s=.5, alpha=.5, c='black')

plt.legend()
plt.xlabel("year")
plt.ylabel("population (mil.)")
plt.subplots_adjust(left=.15, bottom=.20)
plt.savefig("figs/california_texas.pdf")
plt.close()

ipy.embed()
