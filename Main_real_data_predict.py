
from Locate_2D.locator_2d import Locator
from Predictor.predictor import Predictor

import numpy as np
import socket
import time
import can
import cv2
import keyboard as kb
from ultralytics import YOLO



################## Values ##################

MTX = np.array([[1.31527123e+03,0.00000000e+00,5.82870287e+02],
                [0.00000000e+00,1.31458950e+03,5.74049534e+02],
                [0.00000000e+00,0.00000000e+00,1.00000000e+00]])
SHOW = True             #是否展示画面
SYSTEM_SCALING = 1.5
img_width = 1280        #设置相机画面宽高
img_height = 1024

TARGET_HEIGHT = -0.3

SEND_TO_DEVICE = True
DEVICE = 'VMWARE'   #'VMWARE' or 'CAN'
TARGET_SENSOR  = ('127.0.0.1', 5005)   # 调试信息发送地址，发至虚拟机（非核心代码）
TARGET_FILTER  = ('127.0.0.1', 6006)   # 调试信息发送地址，发至虚拟机（非核心代码）

###############################################





################# Properties ##################
model = YOLO(r"D:\XAL\model\exp\mdl_5\weights\best.pt")

locator = Locator(img_width, img_height, radius=0.1043, mtx=MTX)

cam = cv2.VideoCapture(0)

my_bus = can.Bus(channel='COM3', interface='slcan', bitrate=500000)
my_bus.shutdown()

predictor = Predictor(drag_coefficient=0.014, sigma_Q=0.01, mass=0.35, g=9.8, sigma_R=0.1, alpha=0.9, beta=3, kappa=1.7, target_height=TARGET_HEIGHT, que_size=15, threshold_height=0)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #（非核心代码）
###############################################





################ Camera_param #################
cam.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)        #画面宽
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)      #画面高
cam.set(cv2.CAP_PROP_EXPOSURE,-10)                   #曝光时间
cam.set(cv2.CAP_PROP_GAIN, 128)
cam.set(cv2.CAP_PROP_SATURATION, 100)
cam.set(cv2.CAP_PROP_FPS, 200)
cam.set(cv2.CAP_PROP_BRIGHTNESS, 24)
cam.set(cv2.CAP_PROP_CONTRAST, 76)
###############################################










if __name__ == "__main__":


    while True:

        if kb.is_pressed('esc') or kb.is_pressed('backspace'):
            break


        ret,frame = cam.read()
        time_stamp = time.time()

        if ret:

            result = model.predict(frame, classes=[0], conf=0.8, augment=False, verbose=False)
            
            if not len(result[0].boxes) == 0:
                
                boxes = result[0].boxes.xywh.cpu().numpy()
                x, y, w, h = boxes[0]
                ((x1 ,x2), y1, z1) = locator.Locate(center_x=x, center_y=y, box_width=w, box_height=h)
                #获取边框位置并进行粗定位

                predicted_point = predictor.push_get(np.array([min(x1,x2), y1, z1]), time_stamp)
                print(predicted_point)

                if SHOW:
                    cv2.rectangle(frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (120,255,0), 2)
                    #绘制边框

                    cv2.putText(frame, f"x1:{x1}", (int(x-w/2), int(y-h/2-5 )), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [120,255,0], 1)
                    cv2.putText(frame, f"x2:{x2}", (int(x-w/2), int(y-h/2-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [120,255,0], 1)
                    cv2.putText(frame, f"y :{y1}", (int(x-w/2), int(y-h/2-35)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [120,255,0], 1)
                    cv2.putText(frame, f"z :{z1}", (int(x-w/2), int(y-h/2-50)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [120,255,0], 1)
                    #绘制世界坐标信息
                
                    

                if SEND_TO_DEVICE:

                    if DEVICE == 'VMWARE':
                        msg = f"{min(x1,x2):.3f},{y1:.3f},{z1:3f},{time_stamp:.7f}"
                        sock.sendto(msg.encode(), TARGET_SENSOR)

                        if predicted_point is not None:
                            
                            msg = f"{predicted_point[0]:.3f},{predicted_point[1]:.3f},{TARGET_HEIGHT},{time_stamp:.7f}"
                            sock.sendto(msg.encode(), TARGET_FILTER)

                    elif DEVICE =='CAN':
                        pass


            else:
                predictor.push_get(None, None)


            if SHOW:
                cv2.line(frame, (0, img_height//2), (img_width, img_height//2), (255,255,255), 1)
                cv2.line(frame, (img_width//2, 0), (img_width//2, img_height), (255,255,255), 1)
                #十字准星

                cv2.imshow("Cam", cv2.resize(frame, (int(img_width/SYSTEM_SCALING), int(img_height/SYSTEM_SCALING))))
                cv2.waitKey(1) 

            
        else:
            print("Device not connected!")
            break


    
    cv2.destroyAllWindows()
    exit()