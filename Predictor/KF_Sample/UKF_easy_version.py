import numpy as np
import matplotlib.pyplot as plt
import time

plt.rcParams['font.family'] = 'SimHei'
# ==================== 基础设置 ====================
dt = 0.1
steps = 50

# 状态: [位置, 速度]
state_dim = 2

# ==================== 系统模型 ====================
def state_transition(x, dt):
    """状态转移: 位置+速度*时间 + 加速度"""
    pos, vel = x
    acceleration = 0.5  # 恒定加速度
    new_pos = pos + vel * dt + 0.5 * acceleration * dt**2
    new_vel = vel + acceleration * dt
    return np.array([new_pos, new_vel])

def observation_model(x):
    """观测模型: 观测位置"""
    pos, _ = x
    return np.array([pos])  # 简化：直接观测位置

# ==================== UKF核心 ====================
def ukf_filter(estimated_state, P, observed_data):
    """UKF核心算法"""
    n = state_dim
    
    # 1. 生成Sigma点 (在估计值周围选点)
    sigma_points = np.zeros((n, 2*n+1))
    sigma_points[:, 0] = estimated_state
    
    # 使用矩阵平方根分散点
    S = np.linalg.cholesky(P)
    
    for i in range(n):
        sigma_points[:, i+1] = estimated_state + S[i, :]
        sigma_points[:, i+1+n] = estimated_state - S[i, :]
    print("A:",np.cov(estimated_state))
    print("B:",np.mean(np.cov(sigma_points.T)))
    # 2. 预测步 (让所有点按照运动模型移动)
    predicted_points = np.zeros((n, 2*n+1))
    for i in range(2*n+1):
        predicted_points[:, i] = state_transition(sigma_points[:, i], dt)
    
    # 计算预测的均值 (新位置的中心点)
    predicted_state = np.mean(predicted_points, axis=1)
    
    # 计算预测的协方差 (新位置的不确定性)
    predicted_P = np.cov(predicted_points) + np.diag([0.1, 0.1])  # 加过程噪声
    
    # 3. 更新步 (用观测值修正预测)
    # 预测观测值
    predicted_obs = observation_model(predicted_state)
    
    # 计算卡尔曼增益 (信任预测还是信任观测)
    innovation = observed_data - predicted_obs
    K = np.array([0.8, 0.1])  # 简化增益计算
    
    # 状态更新
    estimated_state = predicted_state + K * innovation[0]
    
    # 协方差更新
    P = predicted_P * 0.9  # 简化更新
    
    return estimated_state, P

# ==================== 主程序 ====================
# 初始化
true_state = np.array([0.0, 1.0])      # 真实状态 [位置, 速度]
estimated_state = np.array([0.5, 0.8]) # 估计状态 (初始有误差)
P = np.diag([1.0, 1.0])               # 不确定性矩阵

# 存储结果
true_positions = []
estimated_positions = []
observations = []

print("开始简化UKF仿真...")

for step in range(steps):
    # 1. 真实系统前进
    true_state = state_transition(true_state, dt)
    
    # 2. 生成带噪声的观测值
    true_obs = observation_model(true_state)
    observed_data = true_obs + np.random.normal(0, 0.3)  # 加观测噪声
    
    # 3. UKF滤波
    estimated_state, P = ukf_filter(estimated_state, P, observed_data)
    
    # 存储结果
    true_positions.append(true_state[0])
    estimated_positions.append(estimated_state[0])
    observations.append(observed_data[0])
    
    if step % 10 == 0:
        print(f"步骤 {step}: 真实={true_state[0]:.2f}, 估计={estimated_state[0]:.2f}")

# ==================== 绘图 ====================
plt.figure(figsize=(10, 6))
time_axis = np.arange(steps) * dt

plt.plot(time_axis, true_positions, 'g-', label='真实位置', linewidth=2)
plt.plot(time_axis, estimated_positions, 'b--', label='UKF估计', linewidth=2)
plt.plot(time_axis, observations, 'r.', label='观测值', markersize=4, alpha=0.6)

plt.xlabel('时间 (s)')
plt.ylabel('位置 (m)')
plt.title('UKF滤波 - 位置跟踪')
plt.legend()
plt.grid(True)
plt.show()

print("完成!")