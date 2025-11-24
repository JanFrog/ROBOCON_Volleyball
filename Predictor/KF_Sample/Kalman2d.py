"""
======================================================================
一、场景说明
----------------------------------------------------------------------
- 真实运动：小车在 XY 平面做匀速直线运动（速度恒定，无控制输入）
- 观测方式：模拟 GPS, 每一帧给出带高斯噪声的坐标
- 滤波目标：用卡尔曼滤波从带噪 GPS 里还原出真实轨迹与速度
======================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'SimHei'

# 1. 仿真参数 --------------------------------------------------------------
T  = 0.1              # 采样周期，单位：s
N  = 200              # 总步数
truth_vx, truth_vy = 8, 6   # 真实恒定速度（m/s）
gps_noise_std = 5     # GPS 观测噪声标准差（m）

# 2. 生成“真实”轨迹与带噪观测 ---------------------------------------------
truth_pos = np.zeros((N, 2))   # 真实位置
measurement = np.zeros((N, 2)) # GPS 观测
for k in range(N):
    t = k * T
    truth_pos[k] = [truth_vx * t, truth_vy * t]
    measurement[k] = truth_pos[k] + np.random.normal(0, gps_noise_std, 2)

# 3. 卡尔曼滤波器初始化 -----------------------------------------------------
# 状态向量 x = [x, y, vx, vy]^T
# 观测向量 z = [x_gps, y_gps]^T

# 3-1 状态转移矩阵 F（匀速模型）
F = np.array([[1, 0, T, 0],
              [0, 1, 0, T],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

# 3-2 观测矩阵 H（只观测位置，不观测速度）
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

# 3-3 过程噪声协方差 Q
# 假设加速度是零均值白噪声，协方差强度 q
q = 0.1   # 可看成“对模型信任度”的倒数，调参点

Q = q * np.array([[T**3/3, 0,      T**2/2, 0     ],#模型不确定度
                  [0,      T**3/3, 0     , T**2/2],
                  [T**2/2, 0,     T     , 0     ],
                  [0,      T**2/2, 0     , T     ]])

# 3-4 观测噪声协方差 R
R = gps_noise_std**2 * np.eye(2)

# 3-5 初始状态与协方差
x_hat = np.array([0, 0, 0, 0], dtype=float)  # 初始估计：原点、速度 0
P = np.diag([10, 10, 10, 10])**2               # 初始不确定度要大一点

# 4. 预留容器记录滤波结果 ----------------------------------------------------
kf_pos = np.zeros((N, 2))
kf_vel = np.zeros((N, 2))




# 5. 主循环：卡尔曼滤波 ------------------------------------------------------
for k in range(N):
    z = measurement[k]          # 当前 GPS 观测

    # 5-1 预测步 (Predict)
    x_hat_pred = F @ x_hat###      # 状态预测      （4维）
    P_pred     = F @ P @ F.T + Q  # 协方差预测      （4阶）

    # 5-2 更新步 (Update)
    y = z - H @ x_hat_pred      # 观测残差（新息）  （2维）
    S = H @ P_pred @ H.T + R    # 残差协方差        （2阶）
    K = P_pred @ H.T @ np.linalg.inv(S)  # 卡尔曼增益

    x_hat = x_hat_pred + K @ y#  # 状态修正
    P = (np.eye(4) - K @ H) @ P_pred     # 协方差修正

    # 记录
    kf_pos[k] = x_hat[:2]
    kf_vel[k] = x_hat[2:]











# 6. 可视化结果 --------------------------------------------------------------
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(truth_pos[:, 0], truth_pos[:, 1], 'g-', label='真实轨迹')
plt.plot(measurement[:, 0], measurement[:, 1], 'r+', alpha=0.6, label='GPS 观测')
plt.plot(kf_pos[:, 0], kf_pos[:, 1], 'b-', label='KF 估计')
plt.legend(); plt.title('轨迹对比'); plt.axis('equal')

plt.subplot(1, 2, 2)
plt.plot(np.ones(N)*truth_vx, 'g--', label='真实 vx')
plt.plot(np.ones(N)*truth_vy, 'g--', label='真实 vy')
plt.plot(kf_vel[:, 0], 'b-', label='KF vx')
plt.plot(kf_vel[:, 1], 'r-', label='KF vy')
plt.legend(); plt.title('速度估计'); plt.xlabel('帧数')

plt.tight_layout()
plt.show()