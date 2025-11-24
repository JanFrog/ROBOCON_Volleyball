import cv2
import numpy as np
import os


images_dir = "C:\\Users\\JanFrog(LEGION)\\Desktop\\chessboard_image\\"


# cam = cv2.VideoCapture(1)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)

points_xyz = np.zeros(shape=(9*6,3),dtype=np.float32)
points_xyz[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)


image_points = []
objests_points = []
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for file in os.listdir(images_dir):

    image = cv2.imread(os.path.join(images_dir,file))
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image_gray = cv2.GaussianBlur(image_gray,(3,3),0.5,None)

    chessboard_find,points = cv2.findChessboardCorners(image_gray,(9,6),None)

    if chessboard_find:

        objests_points.append(points_xyz)
        image_points.append(points)
        points2 = cv2.cornerSubPix(image_gray, points, (11,11), (-1,-1), criteria)
        cv2.drawChessboardCorners(image,(9,6),points,True)

    else:
        print(file,"Not Found!")

    cv2.imshow("wow",image)
    cv2.waitKey(50)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objests_points,image_points,image_gray[::-1].shape,None,None)

print(mtx)

angel_x = np.rad2deg(np.arctan(1920/2/mtx[0][0])*2)
angel_y = np.rad2deg(np.arctan(1080/2/mtx[1][1])*2)

print(angel_x,angel_y)

# h,w = image_gray.shape
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# for file in os.listdir(images_dir):
#     image = cv2.imread(os.path.join(images_dir, file))
#     dst = cv2.undistort(image, mtx, dist, None, newcameramtx)
#     cv2.imwrite(f"{os.path.join(images_dir, file)}_calibresult.png", dst)