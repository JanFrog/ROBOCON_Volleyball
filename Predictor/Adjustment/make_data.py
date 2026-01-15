import cv2
import keyboard as kb
import time
import os

opt_dir = r"D:\Code_Projects\RC\Volleyball\Predictor\Adjustment\real_data"

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)        #画面宽
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)      #画面高
cap.set(cv2.CAP_PROP_EXPOSURE,-10)                   #曝光时间
cap.set(cv2.CAP_PROP_GAIN, 128)
cap.set(cv2.CAP_PROP_SATURATION, 100)
cap.set(cv2.CAP_PROP_FPS, 200)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 26)
cap.set(cv2.CAP_PROP_CONTRAST, 76)
cap.set(cv2.CAP_PROP_FPS, 400)

num = 0
state = 0
save_cache = []
now_dir = None
span = None


while True:

    ret, frame = cap.read()
    tick = time.time()

    cv2.imshow("ccb", frame)
    cv2.waitKey(1)
    
    if kb.is_pressed('backspace') or kb.is_pressed('esc'):
        cv2.destroyAllWindows()
        break



    if kb.is_pressed('enter'):

        if state == 0:
            state = 1
            span = tick
            print("start recording!!")
            time.sleep(0.3)

        elif state == 1:
            span = tick - span
            state = 0
            print("stop recording!!")
            time.sleep(0.3)


    if state == 1:
        if num == 0:
            now_dir = os.path.join(opt_dir,f"{time.localtime(tick).tm_year}{time.localtime(tick).tm_mon:0>{2}}{time.localtime(tick).tm_mday:0>{2}}_{time.localtime(tick).tm_hour:0>{2}}h{time.localtime(tick).tm_min:0>{2}}m{time.localtime(tick).tm_sec:0>{2}}s")
            os.mkdir(now_dir)
            

        save_cache.append((frame, tick))
        num += 1

        if num >= 3000:
            state = 0
            print("stop recording!!(out of memories)")
            time.sleep(0.3)



    if state == 0:

        if num > 0:
            cv2.destroyAllWindows()
            for i, (img, timestamp) in enumerate(save_cache):

                print(f"saving: {i:0>6}.png")
                cv2.imwrite(os.path.join(opt_dir, os.path.join(now_dir,f"{i:0>{6}}_{timestamp}.png")), img)

            print("Done!!")

            file = open(os.path.join(now_dir,"info.txt"), "w")
            file.write(f"average frame rate: {round(num/span, 1)}pfs\n")
            file.write(f"MTX: [[1.31527123e+03,0.00000000e+00,5.82870287e+02],\n      [0.00000000e+00,1.31458950e+03,5.74049534e+02],\n      [0.00000000e+00,0.00000000e+00,1.00000000e+00]]\n")
            file.close()

            now_dir = None
            num = 0
            span = None
            save_cache.clear()



exit()