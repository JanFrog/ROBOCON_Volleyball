import numpy as np


class Locator:
    

    def __init__(self,img_width=None,img_height=None,radius=None,mtx=None,angel_width=None,angel_height=None):
        ############################参数说明#############################
        #                  mtx: 相机内参矩阵                             #
        #   angel_width/height: 横/纵向张角(弧度)  *注:不是视场角FOV     #
        #     img_width/height: 画面横/纵向尺寸(像素)                    #
        #               radius: 要检测的球体半径(米)                     #
        #################################################################


        self.img_w = img_width      #画面宽(像素)
        self.img_h = img_height     #画面高(像素)
        self.radius = radius        #目标球真实半径(米)


        if mtx is not None:     #优先使用内参矩阵数据
            self.f1 = mtx[0][0]
            self.f2 = mtx[1][1]


        else:

            self.f1 = (self.img_w/2) / np.tan(angel_width/2)#焦点与屏幕距离(横向像素/mm)
            self.f2 = (self.img_h/2) / np.tan(angel_height/2)#同上(纵向像素/mm)

        print(f"f1:{self.f1}")#输出
        print(f"f2:{self.f2}")#检查

        # self.d_avg = (d1+d2)/2  #焦点与屏幕距离，但是取平均



    def __Count_1d(self,val_1,val_2,f):

        #val_1,val_2分别为球体左右（或上下）边框位置与屏幕中线的差值（像素）

        val_delta = val_2 - val_1    #左右（或上下）边框距离差（像素）

        theta_1 = np.arctan(val_1 / f)     #球体左边缘相对于画面中心偏角（弧度）
        theta_2 = np.arctan(val_2 / f)     #球体右边缘相对于画面中心偏角（弧度）

        beta = (val_1*np.sin(theta_1)) / (val_2*np.sin(theta_2))    #角平分线定理，求得球心与左右（或上下）两边框距离之比
        
        val_center = val_1 + (val_delta*(beta/(1+beta)))    #球心距离画面中心一维距离（像素）
        
        theta_center = np.arctan(val_center/f)     #球心所在射线与中心射线一维偏转角（弧度）

        distance = ((self.radius/np.cos(theta_1))+(self.radius/np.cos(theta_2))) * f / val_delta
        #用相似三角形求得：球心平面与焦点平面距离（米）

        return distance*np.tan(theta_center) , distance



    def Locate(self,center_x,center_y,box_width,box_height):

        #改为以成像中心为心
        valx_1 = center_x - (box_width / 2) - (self.img_w / 2)
        valx_2 = center_x + (box_width/2) - (self.img_w/2)
        valy_1 = center_y - (box_height/2) - (self.img_h/2)
        valy_2 = center_y + (box_height/2) - (self.img_h/2)

        
        cam_y,cam_x1 = self.__Count_1d(valx_1,valx_2,self.f1)
        cam_z,cam_x2 = self.__Count_1d(valy_1,valy_2,self.f2)

        cam_y *= -1
        cam_z *= -1
        # cam_x = (cam_x1 + cam_x2) / 2
        #相机朝前的情况下：向前x,向左y,向上z

        return ((cam_x1,cam_x2),cam_y,cam_z)