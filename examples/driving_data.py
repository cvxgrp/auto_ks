import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import g
from scipy.spatial.transform import Rotation as R
import navpy
import auto_ks
from pykalman import KalmanFilter
import IPython as ipy
import time
from utils import latexify

np.random.seed(0)
np.set_printoptions(precision=1)

# read data
df = pd.read_csv('data/v5-20190819-173113-381s.csv').iloc[2000:20000]
accel = df[['accelUserX(g)', 'accelUserY(g)', 'accelUserZ(g)']]
rpy = df[['Roll(rads)', 'Pitch(rads)', 'Yaw(rads)']]
latlong = np.array(df[['Lat', 'Long']])
speed = df[['Speed(m/s)']]
alt = df[['Alt(m)']]

# preprocess
heading = (np.pi + np.deg2rad(df[['TrueHeading']] - 90)) % (2 * np.pi ) - np.pi
velocity = np.array(speed) * np.hstack([np.cos(-heading), np.sin(-heading)])
rot = R.from_euler('yxz', rpy)
accel = rot.apply(accel) * g
gps_ned = navpy.lla2ned(latlong[:, 0], latlong[:, 1], alt, 37.422704, -122.161435, 0.0)
gps_enu = np.hstack([gps_ned[:, 1][:, None], gps_ned[:, 0][:, None], -gps_ned[:, 2][:, None]])

# concatenate measurements
y = np.hstack([
    gps_enu, accel, velocity
])

# create known measurements mask; gps and velocity measurements are only valid when they change
K_gps = np.append(np.array([True]), np.linalg.norm(np.diff(y[:, :3], axis=0), axis=1) != 0)
K_vel = np.append(np.array([True]), np.linalg.norm(np.diff(y[:, -2:], axis=0), axis=1) != 0)
K = np.hstack([
    np.repeat(K_gps[:, None], 3, axis=1),
    np.ones((K_gps.size, 3), dtype=np.bool),
    np.repeat(K_vel[:, None], 2, axis=1)
])

# initialize parameters
eye = np.eye(3)
zeros = np.zeros((3, 3))
h = 1. / 100.0
lam = 1e-10
alpha = 1e-4
n = 9
m = 8
A = np.bmat([
    [eye, h * eye, zeros],
    [zeros, eye, h * eye],
    [zeros, zeros, eye]
])
C = np.bmat([
    [eye, zeros, zeros],
    [zeros, zeros, eye],
    [np.zeros((2, 3)), np.eye(2), np.zeros((2, 4))]
])
W_neg_sqrt = np.eye(n)
V_neg_sqrt = .01*np.eye(m)
params_initial = auto_ks.KalmanSmootherParameters(np.array(A), np.array(W_neg_sqrt), np.array(C), np.array(V_neg_sqrt))

# proximal operator for r
def prox(params, t):
    r = 0.0
    W_neg_sqrt = params.W_neg_sqrt / (t * alpha + 1.0)
    idx = np.arange(W_neg_sqrt.shape[0])
    W_neg_sqrt[idx, idx] = 0.0
    r += alpha * np.sum(np.square(W_neg_sqrt))
    W_neg_sqrt[idx, idx] = np.diag(params.W_neg_sqrt)

    V_neg_sqrt = params.V_neg_sqrt / (t * alpha + 1.0)
    idx = np.arange(V_neg_sqrt.shape[0])
    V_neg_sqrt[idx, idx] = 0.0
    r += alpha * np.sum(np.square(V_neg_sqrt))
    V_neg_sqrt[idx, idx] = np.diag(params.V_neg_sqrt)

    return auto_ks.KalmanSmootherParameters(A, W_neg_sqrt, C, V_neg_sqrt), r

# choose missing measurements
gps_meas_idx = np.where(K_gps)[0]
np.random.shuffle(gps_meas_idx)
M = np.zeros(y.shape, dtype=np.bool)
M[gps_meas_idx[:int(gps_meas_idx.size * .2)], :3] = True

# choose test measurements
M_test = np.zeros(y.shape, dtype=np.bool)
M_test[gps_meas_idx[int(gps_meas_idx.size * .2):int(gps_meas_idx.size * .4)], :3] = True

# evaluate initial parameters
xhat_initial, yhat_initial, DT = auto_ks.kalman_smoother(params_initial, y, K & ~M_test, lam)
loss_auto = np.linalg.norm(yhat_initial[M_test] - y[M_test])**2 / M_test.sum()
print("initial test loss", loss_auto)

# run kalman smoother auto-tuning
tic = time.time()
params, info = auto_ks.tune(params_initial, prox, y, K & ~M_test, lam, M=M, lr=1e-2, verbose=True, niter=25)
toc = time.time()
print("time", toc-tic, "s")
xhat, yhat, DT = auto_ks.kalman_smoother(params, y, K & ~M_test, lam)

# evaluate
loss_auto = np.linalg.norm(yhat[M_test] - y[M_test])**2 / M_test.sum()
print ("final_test_loss:", loss_auto)

print(np.diag(np.linalg.inv(params.W_neg_sqrt @ params.W_neg_sqrt.T)))
print(np.diag(np.linalg.inv(params.V_neg_sqrt @ params.V_neg_sqrt.T)))

latexify(4, 3)
plt.plot(yhat_initial[::50, 0], yhat_initial[::50, 1], '--', alpha=.5, c='black', label='before')
plt.plot(yhat[::50, 0], yhat[::50, 1], '-', alpha=.5, c='black', label='after')
plt.scatter(y[::50, 0], y[::50, 1], s=.5, c='black')
plt.legend()
plt.xlabel("East (m)")
plt.ylabel("North (m)")
plt.subplots_adjust(left=.15, bottom=.2)
plt.savefig("figs/driving.pdf")
plt.close()

ipy.embed()