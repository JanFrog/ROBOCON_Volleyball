#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2 as pc2
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', 5005))




rospy.init_node('fake_cloud_py')
pub = rospy.Publisher('/fake_cloud', PointCloud2, queue_size=10)

# 定义字段（必须人为写）
fields = [PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
          PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
          PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1)]

while not rospy.is_shutdown():
    
    data, addr = sock.recvfrom(1024)

    x, y, z, t = data.decode().split(',')

    points = [[float(x),float(y),float(z)]]

    header = rospy.Header(stamp=rospy.Time.now(), frame_id='map')
    cloud = pc2.create_cloud(header, fields, points)
    pub.publish(cloud)