
import numpy as np
import socket
import time
import cv2
import os
import sys
import csv
from ultralytics import YOLO

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from Locate_2D.locator_2d import Locator
from Predictor.predictor import Predictor



################## Values ##################

MTX = np.array([[1.31527123e+03,0.00000000e+00,5.82870287e+02],
                [0.00000000e+00,1.31458950e+03,5.74049534e+02],
                [0.00000000e+00,0.00000000e+00,1.00000000e+00]])
SHOW = True             #是否展示画面
SYSTEM_SCALING = 1.5
img_width = 1280        #设置相机画面宽高
img_height = 1024

TARGET_HEIGHT = -0.3

SEND_TO_VMWARE = True
TARGET_SENSOR  = ('127.0.0.1', 5005)   # 调试信息发送地址，发至虚拟机（非核心代码）
TARGET_FILTER  = ('127.0.0.1', 6006)   # 调试信息发送地址，发至虚拟机（非核心代码）

###############################################





################# Properties ##################
model = YOLO(r"D:\XAL\model\exp\mdl_7\weights\best.pt")
data_dir = r"D:\Code_Projects\RC\Volleyball\Predictor\Adjustment\real_data\20260115_16h23m28s"

locator = Locator(img_width, img_height, radius=0.1043, mtx=MTX)
predictor = Predictor(drag_coefficient=0.014, sigma_Q=0.01, mass=0.35, g=9.8, sigma_R=0.1, alpha=0.9, beta=3, kappa=1.7, target_height=TARGET_HEIGHT, que_size=15, threshold_height=0)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #（非核心代码）
###############################################




if __name__ == "__main__":

    try:
        all_img_names = os.listdir(data_dir)
    except:
        print(f'ERROR! Can\'t find that dir: {data_dir}')
        exit()

    try:
        all_img_names.remove("info.txt")
        all_img_names.remove("points.csv")
    except:
        pass


    if os.path.exists(os.path.join(data_dir, "points.csv")):
        os.remove(os.path.join(data_dir, "points.csv"))

    with open(os.path.join(data_dir, "points.csv"), mode='w', encoding='utf-8', newline='') as file:
        
        csv_writer = csv.writer(file)                                           #写入csv
        csv_writer.writerow(['num', 'x(min)', 'y', 'z', 'x_pred', 'y_pred'])



        for i, img_name in enumerate(all_img_names):
            
            frame = cv2.imread(os.path.join(data_dir, img_name))
            time_stamp = float(img_name.rsplit('_', 1)[1].rsplit('.', 1)[0])


            result = model.predict(frame, classes=[0], conf=0.25, augment=False, verbose=False)
            
            if not len(result[0].boxes) == 0:
                
                boxes = result[0].boxes.xywh.cpu().numpy()
                x, y, w, h = boxes[0]
                ((x1 ,x2), y1, z1) = locator.Locate(center_x=x, center_y=y, box_width=w, box_height=h)
                #获取边框位置并进行粗定位


                predicted_point = predictor.push_get(np.array([min(x1,x2), y1, z1]), time_stamp)
                print(predicted_point)

                if not np.any(predicted_point == None):
                    csv_writer.writerow([i, min(x1, x2), y1, z1, predicted_point[0], predicted_point[1]])
                else:
                    csv_writer.writerow([i, min(x1, x2), y1, z1])

                if SHOW:
                    cv2.rectangle(frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (120,255,0), 2)
                    #绘制边框

                    cv2.putText(frame, f"x1:{x1}", (int(x-w/2), int(y-h/2-5 )), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [120,255,0], 1)
                    cv2.putText(frame, f"x2:{x2}", (int(x-w/2), int(y-h/2-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [120,255,0], 1)
                    cv2.putText(frame, f"y :{y1}", (int(x-w/2), int(y-h/2-35)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [120,255,0], 1)
                    cv2.putText(frame, f"z :{z1}", (int(x-w/2), int(y-h/2-50)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [120,255,0], 1)
                    #绘制世界坐标信息
                
                    

                if SEND_TO_VMWARE:
                    msg = f"{min(x1,x2):.3f},{y1:.3f},{z1:3f},{time_stamp:.7f}"
                    sock.sendto(msg.encode(), TARGET_SENSOR)

                    if predicted_point is not None:
                        
                        msg = f"{predicted_point[0]:.3f},{predicted_point[1]:.3f},{TARGET_HEIGHT},{time_stamp:.7f}"
                        sock.sendto(msg.encode(), TARGET_FILTER)

            else:
                predictor.push_get(None, None)
                csv_writer.writerow([i])
                


            if SHOW:
                cv2.line(frame, (0, img_height//2), (img_width, img_height//2), (255,255,255), 1)
                cv2.line(frame, (img_width//2, 0), (img_width//2, img_height), (255,255,255), 1)
                #十字准星

                cv2.imshow("Cam", cv2.resize(frame, (int(img_width/SYSTEM_SCALING), int(img_height/SYSTEM_SCALING))))
                cv2.waitKey(1)