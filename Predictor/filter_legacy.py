
import numpy as np



class UKF:

    def __init__(self, drag_coefficient = 0.014 , g = 9.8, mass = 0.125, sigma_R = 0.1, sigma_Q = 1, alpha = 1, beta = 2, kappa = 0):
        #低速球体阻力系数近似看成0.47, 据此计算k
        #默认状态向量6维, 观测向量3维

        self.K                = drag_coefficient            #阻力系数(F = kv^2 中的k)
        self.mass             = mass                        #排球质量
        self.g                = g                           #重力加速度
        self.R                = (sigma_R**2) * np.eye(3)    #观测噪声协方差矩阵, 0.3是观测噪声标准差
        self.n                = 6                           #状态向量的维度([x, y, z, v_x, v_y, v_z])
        self.m                = 3                           #观测向量的维度([x, y, z ])
        self.sigma_Q          = sigma_Q
        self.character_vel    = np.sqrt(self.mass * self.g / self.K)      #特征速率 (提前计算减少计算量)
        self.traffic          = np.sqrt( self.g * self.K / self.mass)     #特征流量 (提前计算减少计算量)

        #parameters for UT
        self.λ              = alpha ** 2 * (self.n + kappa) - self.n            #计算λ
        self.Weight_M       = np.full(2*self.n+1, 1 / (2 * (self.n + self.λ)))  #均值权重
        self.Weight_M[0]    = self.λ / (self.n + self.λ)                        #中心点修正
        self.Weight_C       = self.Weight_M.copy()                              #协方差权重
        self.Weight_C[0]   += (1 - alpha**2 + beta)                             #中心点修正
        print(f"λ：{self.λ}")
        
    





    def __count_vel_posi_vel(self, vel, dt):
        return self.character_vel * np.tan(np.arctan(vel / self.character_vel) - dt * self.traffic)

    def __count_vel_nega_vel(self, vel, dt):
        if (vel + self.character_vel) != 0:
            return self.character_vel * ((2 / (1 + ((2 * self.character_vel / (vel + self.character_vel) - 1) * (np.e ** (2 * dt * self.traffic))))) - 1)
        else:
            return vel
        
    def _count_pos_posi_vel(self, vel, pos, dt):
        return ((self.mass / self.K) * np.log(np.cos(np.arctan(vel / self.character_vel) - (self.traffic * dt)) / np.cos(np.arctan(vel / self.character_vel)))) + pos

    def _count_pos_nega_vel(self, vel, pos, dt):
        if (vel + self.character_vel) != 0:
            return (self.mass / self.K) * np.log((2 * self.character_vel / (vel + self.character_vel)) / (1 + (2 * self.character_vel / (vel + self.character_vel) - 1) * (np.e ** (2 * dt * self.traffic)))) + pos + (dt * self.character_vel)
        else:
            return self.character_vel * dt + pos
        


    def __count_transition_with_acc(self, pos, vel, dt) -> float:   #单个维度状态转移的计算函数 (有加速度)

        if vel > 0:
            threshold = np.arctan(vel / self.character_vel) / self.traffic

            if dt <= threshold:
                new_vel = self.__count_vel_posi_vel(vel, dt)
                new_pos = self._count_pos_posi_vel(vel, pos, dt)

            else:
                new_vel = self.__count_vel_nega_vel(0, dt-threshold)
                new_pos = self._count_pos_nega_vel(0, self._count_pos_posi_vel(vel, pos, threshold), dt-threshold)

        else:
            new_vel = self.__count_vel_nega_vel(vel, dt)
            new_pos = self._count_pos_nega_vel(vel, pos, dt)
        
        return new_pos, new_vel



    def _count_transition(self, pos, vel, dt):     #单个维度状态转移的计算函数 (无加速度)

        if vel != 0:

            k = self.K * np.sign(vel)    #k与速度同号
            new_pos = pos + (self.mass / k) * np.log(((k * dt *vel / self.mass) + 1))
            new_vel = vel / (((k * dt * vel) / self.mass) + 1)
            return new_pos, new_vel
        
        else:

            return pos, vel



    def __transition(self, point_previous, dt):     #状态转移函数

        result = []

        for pos_x, pos_y, pos_z, vel_x, vel_y, vel_z in point_previous.T:

            pos_x_new, vel_x_new = self._count_transition(pos_x, vel_x, dt)
            pos_y_new, vel_y_new = self._count_transition(pos_y, vel_y, dt)
            pos_z_new, vel_z_new = self.__count_transition_with_acc(pos_z, vel_z, dt)
            result.append(np.array([pos_x_new, pos_y_new, pos_z_new, vel_x_new, vel_y_new, vel_z_new]))
        
        return np.array(result).T
    


    def __sigma_point_generate(self, point, S):

        sigma_points = np.zeros((self.n, 2 * self.n+1))
        sigma_points[:, 0] = point

        for i in range(self.n):
            sigma_points[:, i+1] = point + S[:, i]
            sigma_points[:, i+1+self.n] = point - S[:, i]

        return sigma_points



    def __make_Q(self, dt):

        Q = (self.sigma_Q**2) * \
            np.array([[dt**3/3 ,0       ,0       ,dt**2/2 ,0       ,0       ],
                      [0       ,dt**3/3 ,0       ,0       ,dt**2/2 ,0       ],
                      [0       ,0       ,dt**3/3 ,0       ,0       ,dt**2/2 ],
                      [dt**2/2 ,0       ,0       ,dt      ,0       ,0       ],
                      [0       ,dt**2/2 ,0       ,0       ,dt      ,0       ],
                      [0       ,0       ,dt**2/2 ,0       ,0       ,dt      ]])
        #Q为过程噪声协方差，前面的sigma_a系数为调参点

        return Q
        
    

    def __observe(self, x):
        return np.array([x[i] for i in range(self.m)])
    

    
    def forward(self, estimated_point, P, observed_point, dt): #滤波器核心
       
        try:
            S = np.linalg.cholesky((self.n + self.λ) * P)   #1e-9 是为了防止除0
        
        except:
            print((self.n + self.λ) * P)
            print("ERROR Cholesky!")
            return None
        
        else:

            sigma_points      = self.__sigma_point_generate(estimated_point, S)
            sigma_predict     = self.__transition(sigma_points, dt)
            sigma_predict_obs = self.__observe(sigma_predict)

            predict_P         = np.cov(sigma_predict, rowvar=True) + self.__make_Q(dt) #状态空间の预测协方差
            predict_point     = sigma_predict @ self.Weight_M.T
            predict_point_obs = self.__observe(predict_point)

            Res = observed_point - predict_point_obs    #计算预测与观测的残差



    ############
            # Cov_zz = np.cov(sigma_predict_obs, rowvar=True) + self.R    #观测空间の预测协方差

            # Cov_xz = np.zeros((self.n, self.m))            #计算互协方差

            # for i in range(2 * self.n + 1):
            #     dx = sigma_predict[:, i] - predict_point
            #     dz = sigma_predict_obs[:, i] - predict_point_obs
            #     Cov_xz += np.outer(dx, dz)

            # Cov_xz /= (2 * self.n + 1)
    ############


    ############
    
            Cov_zz = self.Weight_C[0] * np.outer(sigma_predict_obs[:,0] - predict_point_obs,
                                sigma_predict_obs[:,0] - predict_point_obs)
            
            for i in range(1, 2*self.n + 1):
                Cov_zz += self.Weight_C[i] * np.outer(sigma_predict_obs[:,i] - predict_point_obs,
                                            sigma_predict_obs[:,i] - predict_point_obs)
                
            Cov_zz += self.R          # 加测量噪声

            # 2. 互协方差 Pxz  （加权版）
            Cov_xz = self.Weight_C[0] * np.outer(sigma_predict[:,0] - predict_point,
                                        sigma_predict_obs[:,0] - predict_point_obs)
            
            for i in range(1, 2*self.n + 1):
                Cov_xz += self.Weight_C[i] * np.outer(sigma_predict[:,i] - predict_point,
                                            sigma_predict_obs[:,i] - predict_point_obs)
                
    ############




            K = Cov_xz @ np.linalg.inv(Cov_zz)    #卡尔曼增益系数 (Pxz @ Pzz^-1)


            x_updated = predict_point + K @ Res     #更新状态
            P_updated = predict_P - K @ Cov_zz @ K.T   #更新
            
            return x_updated, P_updated