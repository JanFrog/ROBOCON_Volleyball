#!/usr/bin/env python3
"""
2-D projectile tracking with unknown drag coefficient (β) using EKF
Author : you
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------- 参数区 ----------------
g = 9.81            # 重力加速度
dt = 0.02           # 采样周期 (s)
N = 400             # 总步数
q = 0.5             # 过程噪声强度 (m/s²)
r = 3.0             # 观测噪声标准差 (m)

# 真实初始状态  [x, y, vx, vy, β]
x0_true = np.array([0., 0., 50., 40., 0.08])

# 滤波器初始估计
x_est = np.array([0., 0., 45., 35., 0.05])   # 故意给偏差
P0 = np.diag([1., 1., 10., 10., 0.01])       # 初始协方差

# 过程噪声协方差 (只作用在加速度)
Q = np.eye(2) * (q**2)
# 观测噪声协方差
R = np.eye(2) * (r**2)
# ---------------- 结束 ------------------

def true_dynamics(x):
    """无噪声真实一步积分（4阶 RK）"""
    def f(x_):
        v = np.linalg.norm(x_[2:4])
        β = x_[4]
        ax = -β * v * x_[2]
        ay = -g - β * v * x_[3]
        return np.array([x_[2], x_[3], ax, ay, 0.])
    k1 = f(x)
    k2 = f(x + 0.5*dt*k1)
    k3 = f(x + 0.5*dt*k2)
    k4 = f(x + dt*k3)
    return x + dt*(k1 + 2*k2 + 2*k3 + k4)/6.

def ekf_predict(x, P):
    """EKF 预测步，返回 xbar, Pbar, F, G"""
    v = np.linalg.norm(x[2:4])
    β = x[4]
    ax = -β * v * x[2]
    ay = -g - β * v * x[3]

    # 状态转移矩阵  F = df/dx
    F = np.eye(5)
    F[0,2] = dt
    F[1,3] = dt
    # 对 vx,vy 的偏导
    if v > 1e-6:
        dv_dvx = x[2]/v; dv_dvy = x[3]/v
    else:
        dv_dvx = dv_dvy = 0.
    dax_dvx = -β*(v + x[2]*dv_dvx)
    dax_dvy = -β*x[2]*dv_dvy
    day_dvx = -β*x[3]*dv_dvx
    day_dvy = -β*(v + x[3]*dv_dvy)
    F[2,2] += dt*dax_dvx
    F[2,3] += dt*dax_dvy
    F[3,2] += dt*day_dvx
    F[3,3] += dt*day_dvy
    # 对 β 的偏导
    F[2,4] += dt*(-v*x[2])
    F[3,4] += dt*(-v*x[3])

    # 过程噪声雅可比  G = df/dw
    G = np.zeros((5,2))
    G[2,0] = dt
    G[3,1] = dt

    # 状态预测（欧拉即可）
    x_next = x.copy()
    x_next[0] += x[2]*dt
    x_next[1] += x[3]*dt
    x_next[2] += ax*dt
    x_next[3] += ay*dt
    # β 不变

    P_next = F @ P @ F.T + G @ Q @ G.T
    return x_next, P_next, F, G

def ekf_update(xbar, Pbar, z):
    """EKF 更新步，z=[zx,zy]"""
    H = np.array([[1,0,0,0,0],
                  [0,1,0,0,0]])
    y = z - H @ xbar
    S = H @ Pbar @ H.T + R
    K = Pbar @ H.T @ np.linalg.inv(S)
    x = xbar + K @ y
    P = (np.eye(5) - K @ H) @ Pbar
    return x, P

# ---------------- 主循环 -----------------
x_true = x0_true.copy()
x_est  = x_est.copy()
P      = P0.copy()

truth = np.zeros((N,5))
est   = np.zeros((N,5))
meas  = np.zeros((N,2))

for k in range(N):
    # 1. 真值演化
    x_true = true_dynamics(x_true)
    truth[k] = x_true

    # 2. 生成观测
    z = x_true[:2] + np.random.normal(0, r, 2)
    meas[k] = z

    # 3. EKF
    xbar, Pbar, _, _ = ekf_predict(x_est, P)
    x_est, P = ekf_update(xbar, Pbar, z)
    est[k] = x_est

# ---------------- 画图 -------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(truth[:,0], truth[:,1], label='truth')
plt.plot(meas[:,0], meas[:,1], '.', alpha=0.4, label='measurement')
plt.plot(est[:,0], est[:,1], label='EKF')
plt.axis('equal'); plt.legend(); plt.title('Trajectory')

plt.subplot(1,2,2)
plt.plot(truth[:,4]*np.ones(N), 'k', label='true β')
plt.plot(est[:,4], label='est β')
plt.legend(); plt.title('Drag coefficient β')

plt.tight_layout()
plt.show()