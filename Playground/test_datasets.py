import rosbag
from rospy import Time
import datetime
import cv2 as cv
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import pandas as pd


VIODE_msg = ['/cam0/image_raw', '/cam0/segmentation', '/cam1/image_raw', '/cam1/segmentation', '/imu0', '/odometry']
TUMVI_msg = ['/cam0/image_raw', '/cam1/image_raw', '/imu0', '/vrpn_client/raw_transform']

path = "/home/hoangqc/Datasets/TUM_VI/dataset-corridor1_512_16.bag"
path_VIODE = "/home/hoangqc/Datasets/VIODE/city_day/0_none.bag"

ros_msg = TUMVI_msg
bridge = CvBridge()

bag = rosbag.Bag(path)
img0 = img1 = np.zeros([512,512], dtype=np.uint16)
img_prv = np.zeros([512,1024], dtype=np.uint16)

t_start = bag.get_start_time()
s_img0 = bag.get_message_count(topic_filters=[ros_msg[0]])
s_img1 = bag.get_message_count(topic_filters=[ros_msg[1]])

ros_t = []

for topic, msg, t in bag.read_messages(topics=[ros_msg[0]]):
    ros_t.append(t.to_nsec())
    print(t)
    # ...

# pd.DataFrame(np.array(ros_t)).to_csv("img_timestamp.csv")
#
# for i in range(int(s_img0)):
#     for topic, msg, t in bag.read_messages(topics=ros_msg, start_time=ros_t[i], end_time=ros_t[i+1]):
#         if topic == ros_msg[0]:
#             img_prv[:, 0:512] = bridge.imgmsg_to_cv2(msg, "mono16")
#         if topic == ros_msg[1]:
#             img_prv[:, 512:1024] = bridge.imgmsg_to_cv2(msg, "mono16")
#         if topic == ros_msg[2]:
#             print(msg.linear_acceleration)
#             # print(msg.angular_velocity)
#
#
#     cv.imshow("***", img_prv)
#     if cv.waitKey(10) == 27: break
