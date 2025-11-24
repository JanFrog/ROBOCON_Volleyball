
from Locate_2D.locator_2d import Locator
from Predictor.filter import UKF

import numpy as np
import socket
import time
import cv2
from ultralytics import YOLO



################## Values ##################
MTX = np.array([[1.03969610e+03,0.00000000e+00,9.85433753e+02],     #内参矩阵
                [0.00000000e+00,1.02686241e+03,4.92983435e+02],
                [0.00000000e+00,0.00000000e+00,1.00000000e+00]])
SHOW = True             #是否展示画面
SYSTEM_SCALING = 1.5
img_width = 1920        #设置相机画面宽高
img_height = 1080

SEND_TO_VMWARE = True
TARGET = ('127.0.0.1', 5005)   # 调试信息发送地址，发至虚拟机（非核心代码）
###############################################





################# Properties ##################
model = YOLO(".\\Locate_2D\\model\\YOLO11_Liu_Own3_200.pt")
locator = Locator(img_width, img_height, radius=0.12, mtx=MTX)
cam = cv2.VideoCapture(1)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #（非核心代码）
###############################################





################ Camera_param #################
cam.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)        #画面宽
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)      #画面高
cam.set(cv2.CAP_PROP_EXPOSURE,-7)                   #曝光时间
cam.set(cv2.CAP_PROP_BRIGHTNESS,0.5)               #感光度(默认为0.0)
# cam.set(cv2.CAP_PROP_AUTO_EXPOSURE,1)             #恢复默认曝光时间(临时代码)
###############################################










if __name__ == "__main__":


    while True:

        ret,frame = cam.read()
        time_tick = time.time()

        if ret:

            result = model.predict(frame, classes=[0], conf=0.5, augment=False, verbose=False)
            
            if not len(result[0].boxes) == 0:
                
                boxes=result[0].boxes.xywh.cpu().numpy()
                x, y, w, h = boxes[0]
                ((x1 ,x2), y1, z1) = locator.Locate(center_x=x, center_y=y, box_width=w, box_height=h)
                #获取边框位置并进行粗定位

                if SHOW:
                    cv2.rectangle(frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (120,255,0), 2)
                    #绘制边框

                    cv2.putText(frame, f"x1:{x1}", (int(x-w/2), int(y-h/2-5 )), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [120,255,0], 1)
                    cv2.putText(frame, f"x2:{x2}", (int(x-w/2), int(y-h/2-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [120,255,0], 1)
                    cv2.putText(frame, f"y :{y1}", (int(x-w/2), int(y-h/2-35)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [120,255,0], 1)
                    cv2.putText(frame, f"z :{z1}", (int(x-w/2), int(y-h/2-50)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [120,255,0], 1)
                    #绘制世界坐标信息

                    

                if SEND_TO_VMWARE:
                    msg = f"{min(x1,x2):.3f},{y1:.3f},{z1:3f},{time_tick:.7f}"
                    sock.sendto(msg.encode(), TARGET)


            if SHOW:
                cv2.line(frame, (0, img_height//2), (img_width, img_height//2), (255,255,255), 1)
                cv2.line(frame, (img_width//2, 0), (img_width//2, img_height), (255,255,255), 1)
                #十字准星

                cv2.imshow("Cam", cv2.resize(frame, (int(img_width/SYSTEM_SCALING), int(img_height/SYSTEM_SCALING))))
                cv2.waitKey(1) 

            
        else:
            print("Device not connected!")
            break