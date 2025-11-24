import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'SimHei'
# ---------------- UKF 本体 ----------------
class UKF:
    def __init__(self, drag=0.5, g=-9.8, mass=0.125, n=6, m=3):
        self.b, self.g, self.m = drag, g, mass
        self.n, self.m = n, m

    # 私有：生成 sigma 点
    def _sigma(self, x, P, kappa=0):
        lam = self.n + kappa
        sqrtP = np.linalg.cholesky((self.n + lam) * P)
        X = np.zeros((self.n, 2 * self.n + 1))
        X[:, 0] = x
        for i in range(self.n):
            X[:, i + 1] = x + sqrtP[i]
            X[:, i + 1 + self.n] = x - sqrtP[i]
        w = np.full(2 * self.n + 1, 1.0 / (2 * (self.n + lam)))
        w[0] = lam / (self.n + lam)
        wc = w.copy(); wc[0] += (1 - lam + kappa)
        return X, w, wc

    # 状态转移（Euler 积分，无奇异）
    def transition(self, x, dt):
        p, v = x[:3], x[3:]
        k = -self.b * np.sign(v)
        a = k * v / self.m + np.array([0, 0, self.g])
        v_new = v + a * dt
        p_new = p + v_new * dt          # 用新速度
        return np.concatenate([p_new, v_new])

    # 观测
    def observe(self, x):
        return x[:self.m]

    # 一步滤波
    def filter(self, xest, Pest, z, Q, R, dt):
        X, wm, wc = self._sigma(xest, Pest)
        Xp = np.zeros_like(X)
        for i in range(X.shape[1]):
            Xp[:, i] = self.transition(X[:, i], dt)
        xp = Xp @ wm
        Pp = Q.copy()
        for i in range(Xp.shape[1]):
            dx = Xp[:, i] - xp
            Pp += wc[i] * np.outer(dx, dx)

        Zp = np.zeros((self.m, Xp.shape[1]))
        for i in range(Xp.shape[1]):
            Zp[:, i] = self.observe(Xp[:, i])
        zp = Zp @ wm

        Pzz = R.copy()
        Pxz = np.zeros((self.n, self.m))
        for i in range(Zp.shape[1]):
            dz = Zp[:, i] - zp
            Pzz += wc[i] * np.outer(dz, dz)
            dx = Xp[:, i] - xp
            Pxz += wc[i] * np.outer(dx, dz)

        K = np.linalg.solve(Pzz.T, Pxz.T).T
        xnew = xp + K @ (z - zp)
        Pnew = Pp - K @ Pzz @ K.T
        return xnew, Pnew


# ---------------- 仿真真值 ----------------
def true_dynamics(x, dt, drag, g):
    p, v = x[:3], x[3:]
    k = -drag * np.sign(v)
    a = k * v / 0.125 + np.array([0, 0, g])
    v += a * dt
    p += v * dt
    return np.concatenate([p, v])


# ---------------- 主流程 ----------------
dt = 0.002
T = 2000                      # 步数
drag_true = 0.5
ukf = UKF(drag=drag_true)

# 初始状态 [x,y,z,vx,vy,vz]
x0 = np.array([0., 0., 0., 10., 8., 12.])
P0 = np.diag([1e-2]*3 + [1e-1]*3)
Q = np.diag([1e-4]*3 + [1e-3]*3)   # 过程噪声
R = np.diag([5e-2]*3)              # 观测噪声

truth = np.zeros((T, 6))
obs   = np.zeros((T, 3))
est   = np.zeros((T, 6))
x_true = x0.copy()
x_est  = x0.copy()
P_est  = P0.copy()

for k in range(T):
    # 真值 + 观测
    x_true = true_dynamics(x_true, dt, drag_true, -9.8)
    z = x_true[:3] + np.random.multivariate_normal(np.zeros(3), R)
    # UKF 一步
    x_est, P_est = ukf.filter(x_est, P_est, z, Q, R, dt)
    # 记录
    truth[k] = x_true
    obs[k]   = z
    est[k]   = x_est

# ---------------- 画图 ----------------
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(121, projection='3d')
ax.plot(truth[:,0], truth[:,1], truth[:,2], label='Truth')
ax.plot(obs[:,0], obs[:,1], obs[:,2], '.', alpha=0.6, label='Observed')
ax.plot(est[:,0], est[:,1], est[:,2], label='UKF')
ax.set_title('3-D trajectory')

ax2 = fig.add_subplot(122)
err = np.linalg.norm(truth[:,:3] - est[:,:3], axis=1)
ax2.plot(np.arange(T)*dt, err)
ax2.set_xlabel('time / s')
ax2.set_ylabel('position error / m')
ax2.set_title('Estimation error')
plt.tight_layout()
plt.show()