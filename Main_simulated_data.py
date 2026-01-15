from Simulator.simulator import points_generator
from Predictor.filter_legacy import UKF

import numpy as np
import time
import random as rd
import socket


TARGET_SENSOR  = ('127.0.0.1', 5005)   # 调试信息发送地址，发至虚拟机（非核心代码）
TARGET_FILTER  = ('127.0.0.1', 6006)
TARGET_PREDICT = ('127.0.0.1', 7007)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)



pg = points_generator(K=0.014, mass=0.125, g=9.8)
ukf = UKF(drag_coefficient=0.014, sigma_Q=0.07, sigma_R=0.15, alpha=0.9, beta=3, kappa=1.7)





if __name__ == "__main__":

    pg.set_state(-3,-3,0,3.5,3.5,9)
    # pg.set_state_randomly(False, False, False, True, True, True)

    P = np.eye(6) * (5 ** 2)
    state_filtered = np.array([0,0,0,0,0,0])

    while pg.state[2] >= 0:
        
        t0 = time.time()

        # dt = rd.gauss(0.03,0.01)
        dt = 0.05
        pg.update(dt)
        state = pg.get_state(add_noise=False, sigma= 0.2)
        
        try:
            state_filtered, P = ukf.forward(state_filtered, P, pg.get_state(add_noise=True, sigma= 0.15)[:3], dt)
        except:
            print("Cholesky process ERROR !")
            break
        
        msg_1 = f"{state[0]:.3f},{state[1]:.3f},{state[2]:3f},{dt:.7f}"
        msg_2 = f"{state_filtered[0]:.3f},{state_filtered[1]:.3f},{state_filtered[2]:3f},{dt:.7f}"

        print(state_filtered)
        sock.sendto(msg_1.encode(), TARGET_SENSOR)
        sock.sendto(msg_2.encode(), TARGET_FILTER)

        t1 =time.time()

        time.sleep(dt+t1-t0)
        # print(t1-t0)