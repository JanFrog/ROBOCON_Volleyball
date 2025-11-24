import numpy as np
import random as rd




#状态转移方程：https://www.desmos.com/calculator/terbfouttu

class points_generator:
    
    def __init__(self, K = None, mass = None, g = 9.8):
        self.K      = K
        self. mass  = mass
        self.g      = g
        self.state  = np.array([None, None, None, None, None, None])

        self.character_vel  = np.sqrt(self.mass * self.g / self.K)      #特征速度 (提前计算减少计算量)
        self.traffic        = np.sqrt( self.g * self.K / self.mass)     #特征流量 (提前计算减少计算量)
    





    def __count_vel_posi_vel(self, vel, dt):
        return self.character_vel * np.tan(np.arctan(vel / self.character_vel) - dt * self.traffic)

    def __count_vel_nega_vel(self, vel, dt):
        if (vel + self.character_vel) != 0:
            return self.character_vel * ((2 / (1 + ((2 * self.character_vel / (vel + self.character_vel) - 1) * (np.e ** (2 * dt * self.traffic))))) - 1)
        else:
            return vel
        
    def __count_pos_posi_vel(self, vel, pos, dt):
        return ((self.mass / self.K) * np.log(np.cos(np.arctan(vel / self.character_vel) - (self.traffic * dt)) / np.cos(np.arctan(vel / self.character_vel)))) + pos

    def __count_pos_nega_vel(self, vel, pos, dt):
        if (vel + self.character_vel) != 0:
            return (self.mass / self.K) * np.log((2 * self.character_vel / (vel + self.character_vel)) / (1 + (2 * self.character_vel / (vel + self.character_vel) - 1) * (np.e ** (2 * dt * self.traffic)))) + pos + (dt * self.character_vel)
        else:
            return self.character_vel * dt + pos
        


    def __trans_1d_count_with_acc(self, pos, vel, dt) -> float:
        if vel > 0:
            threshold = np.arctan(vel / self.character_vel) / np.sqrt(self.traffic)
            if dt <= threshold:
                new_vel = self.__count_vel_posi_vel(vel, dt)
                new_pos = self.__count_pos_posi_vel(vel, pos, dt)
            else:
                new_vel = self.__count_vel_nega_vel(0, dt-threshold)
                new_pos = self.__count_pos_nega_vel(0, self.__count_pos_posi_vel(vel, pos, threshold), dt-threshold)
        else:
            new_vel = self.__count_vel_nega_vel(vel, dt)
            new_pos = self.__count_pos_nega_vel(vel, pos, dt)
        
        return new_pos, new_vel



    def __trans_1d_count(self, pos, vel, dt) -> float:     #单个维度状态转移的计算函数

        if vel != 0:
            k = self.K * np.sign(vel)    #k与速度同号

            new_pos = pos + (self.mass / k) * np.log(((k * dt *vel / self.mass) + 1))
            new_vel = vel / (((k * dt * vel) / self.mass) + 1)

            return new_pos, new_vel
        
        else:
            return pos, vel
        






    def __trans(self, state_old, dt):

        if dt != 0:
            pos_x, pos_y, pos_z, vel_x, vel_y, vel_z = state_old

            pos_x_new, vel_x_new = self.__trans_1d_count(pos_x, vel_x, dt = dt)
            pos_y_new, vel_y_new = self.__trans_1d_count(pos_y, vel_y, dt = dt)
            pos_z_new, vel_z_new = self.__trans_1d_count_with_acc(pos_z, vel_z, dt = dt)

            return np.array([pos_x_new, pos_y_new, pos_z_new, vel_x_new, vel_y_new, vel_z_new])
        
        else:
            return state_old
    


    def set_state(self, x, y, z, vel_x, vel_y, vel_z):

        if  np.any(self.state == None):
            print("BALL's state inited!")
        else:
            print("BALL's state covered!")

        self.state = np.array([x, y, z, vel_x, vel_y, vel_z])
        return self.state



    def set_state_randomly(self, x=True, y=True, z=True, v_x=True, v_y=True, v_z=True):

        if np.any(self.state == None):
            print("BALL's state inited!")
        else:
            print("BALL's state covered!")

        if x:
            self.state[0] = rd.uniform(-4,4)
        else:
            self.state[0] = 0
        if y:
            self.state[1] = rd.uniform(-4,4)
        else:
            self.state[1] = 0
        if z:
            self.state[2] = rd.uniform(0.1,4)
        else:
            self.state[2] = 0.1
        if v_x:
            self.state[3] = rd.uniform(-4,4)
        else:
            self.state[3] = 3
        if v_y:
            self.state[4] = rd.uniform(-4,4)
        else:
            self.state[4] = 3
        if v_z:
            self.state[5] = rd.uniform(3,5)
        else:
            self.state[5] = 4

        return self.state
    


    def get_state(self, add_noise=False, sigma=0.01):

        if add_noise:
            return self.state + np.array([rd.gauss(sigma=sigma),
                                          rd.gauss(sigma=sigma),
                                          rd.gauss(sigma=sigma),
                                          0,0,0])
        else:
            return self.state



    def update(self, dt):
        if np.all(self.state == None):
            print("ERROR: state not set yet!")
            return None

        elif np.any(self.state == None):
            print("ERROR: state not set properly!")
            return None
        
        else:
            self.state = self.__trans(self.state, dt=dt)
            return self.state