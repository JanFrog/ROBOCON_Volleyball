from Predictor.filter_legacy import UKF
import numpy as np
from collections import deque


class Predictor(UKF):
    def __init__(self, drag_coefficient = 0.014, g = 9.8,
                 mass=0.125, sigma_R = 0.1,
                 sigma_Q = 1, alpha = 0.9,
                 beta = 2, kappa = 0,
                 que_size = 4,
                 target_height = 0,
                 threshold_height = 0):
        
        super().__init__(drag_coefficient, g, mass, sigma_R, sigma_Q, alpha, beta, kappa)

        self.que_size       = que_size
        self.target_height  = target_height
        self.que            = deque([None] * que_size, maxlen = que_size)
        self.th_height      = threshold_height
        self.sequence_count = 0
        self.state     = None
        self.P         = None
        self.last_tick = None



    def __target_count_legacy(self, state):

        if state[5] > 0:

            threshold = np.arctan(state[5] / self.character_vel) / self.traffic

            top_pos = self._count_pos_posi_vel(state[5], state[2], threshold)

            if top_pos >= self.target_height:

                B  = 2 * (np.e ** ((self.K / self.mass) * (top_pos - self.target_height)))
                dt = threshold + (1 / self.traffic) * np.log((B + np.sqrt(B ** 2 - 4)) / 2)

                x, _ = self._count_transition(state[0], state[3], dt)
                y, _ = self._count_transition(state[1], state[4], dt)

                return np.array([x, y])
            
            else:
                return None

        else:

            if state[2] >= self.target_height:

                A = 2 * self.character_vel / (state[5] + self.character_vel) - 1
                B = (A + 1) * (np.e ** ((self.K / self.mass) * (state[2] - self.target_height)))
                dt = (1 / self.traffic) * np.log((B + np.sqrt((B ** 2) - (4 * A))) / (2 * A))

                x, _ = self._count_transition(state[0], state[3], dt)
                y, _ = self._count_transition(state[1], state[4], dt)

                return np.array([x, y])
            
            else:
                return





    def push_get(self, obs: np.ndarray, tick):

        if obs is None:
            if self.sequence_count > 0:
                self.sequence_count -= 1
            return None
        


        elif obs[2] < self.th_height:
            if self.sequence_count > 0:
                self.sequence_count -= 1
            return None

            

        else:

            if self.sequence_count == 0:
                self.P = np.eye(6) * (10 ** 2)
                self.state = np.array([0,0,0,0,0,0])
                self.last_tick = tick - 0.1
            
            self.sequence_count = 3 

            self.state, self.P = self.forward(self.state, self.P, obs, tick-self.last_tick)
            self.last_tick = tick



            # 滤波完求落点解：
            self.que.append(self.__target_count_legacy(self.state))

            if not any(x is None for x in self.que):
                return np.mean(self.que, axis = 0)
            
            else:
                return None















        # self.observe_que.append((coord, timestamp))

        # if not None in self.observe_que:

        #     ((x, y, z), _) = self.observe_que[0]
        #     predict_que = []
        #     xyz = np.array([[x,y,z,0,0,0]])

        #     P = np.eye(6) * (5 ** 2)
        #     for i in range(self.que_size-1):

                
        #         result = self.forward(xyz, P, np.array(self.observe_que[i+1][0]), self.observe_que[i+1][1] - self.observe_que[i][1])
        #         if result == None:
        #             return None
        #         else:
        #             (xyz, P) = result
                
        #         prepoint = self.__target_count(xyz)

        #         if i + 1 >= self.pre_size and prepoint is not None:
        #             predict_que.append(prepoint)

        #     if predict_que:
        #         return np.mean(predict_que, axis = 0)
        #     else:
        #         return None
            
        # else:
        #     return None

            

