import numpy as np
import cv2


img_width = 1920        #设置相机画面宽高
img_height = 1080

cam = cv2.VideoCapture(1)

################ Camera_param #################
cam.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)        #画面宽
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)      #画面高
cam.set(cv2.CAP_PROP_EXPOSURE,-7)                   #曝光时间
cam.set(cv2.CAP_PROP_BRIGHTNESS,0.5)               #感光度(默认为0.0)
# cam.set(cv2.CAP_PROP_AUTO_EXPOSURE,1)             #恢复默认曝光时间(临时代码)
###############################################



if __name__ == "__main__":

    if not cam.isOpened():
        print("无法打摄像机")
        exit()

    else:
        while True:
            break