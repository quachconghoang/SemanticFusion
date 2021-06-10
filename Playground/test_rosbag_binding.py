import rospy
from rospy import Time
import rosbag
from sensor_msgs.msg import Imu, Image, CompressedImage
from geometry_msgs.msg import Pose, PoseStamped, PoseArray, Quaternion
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
import cv2

delta_imu = 0.005 #1/200
delta_img = 0.05 #1/20

rospy.init_node('writer', anonymous=True)
bridge = CvBridge()
ctime = Time.now()
ctime_sec = ctime.to_sec()

bag = rosbag.Bag('/home/hoangqc/Datasets/test.bag', 'w')

for i in range(200):
    m_Imu = Imu()
    m_Pose = Pose()
    newTime = Time.from_sec(ctime_sec + i*delta_imu)
    bag.write(topic='/imu0', msg=m_Imu, t = newTime)
    bag.write(topic='/odometry', msg=m_Pose, t=newTime)
    ...

for i in range(20):
    newTime = Time.from_sec(ctime_sec + i * delta_img)

    img_L = np.zeros((400, 640, 3), np.uint8)
    img_L_depth = np.zeros((400, 640), np.uint16)
    img_R = np.zeros((400, 640, 3), np.uint8)

    img_L.fill(128)
    img_L_depth.fill(32768)
    img_R.fill(255)

    msg = CompressedImage()
    msg.header.stamp = newTime
    msg.format = "jpeg"
    msg.data = np.array(cv2.imencode('.jpg', img_L)[1]).tostring()

    # msg_L = bridge.cv2_to_imgmsg(cvim=img_L, encoding="bgr8")
    # msg_R = bridge.cv2_to_imgmsg(cvim=img_R, encoding="bgr8")

    bag.write(topic='/cam0/image_raw', msg=msg, t=newTime)
    bag.write(topic='/cam1/image_raw', msg=msg, t=newTime)

    ...

bag.close()