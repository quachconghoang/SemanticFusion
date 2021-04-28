import rosbag
from rospy import Time
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path

TUMVI_msg = ['/cam0/image_raw',
             '/cam1/image_raw',
             '/imu0',
             '/vrpn_client/raw_transform']
TUMVI_path = str(Path.home())+ "/Datasets/TUM_VI/dataset-corridor1_512_16.bag"

img_w = 512
img_h = 512

ros_msg = TUMVI_msg
bridge = CvBridge()
# Initiate ORB detector
orb = cv.ORB_create()

bag = rosbag.Bag(TUMVI_path)
img_prv = np.zeros([img_h,img_w*2], dtype=np.uint16)

t_start = bag.get_start_time()
s_img0 = bag.get_message_count(topic_filters=[ros_msg[0]])
# s_img1 = bag.get_message_count(topic_filters=[ros_msg[1]])

ros_t = []

for topic, msg, t in bag.read_messages(topics=[ros_msg[0]]):
    ros_t.append(t)
    # ros_t.append(t.to_nsec())

# pd.DataFrame(np.array(ros_t)).to_csv("tumvi_timestamp.csv")

# i = 50
# st = Time(ros_t[i].secs, ros_t[i].nsecs - 1e6)
# if i > 0: st = Time(ros_t[i - 1].secs, ros_t[i - 1].nsecs + 1e6)
# et = Time(ros_t[i].secs, ros_t[i].nsecs)
#
# for topic, msg, t in bag.read_messages(topics=ros_msg, start_time=st, end_time=et):
#     if topic == ros_msg[0]:
#         img_prv[:, :img_w] = bridge.imgmsg_to_cv2(msg, "mono16")
#     if topic == ros_msg[1]:
#         img_prv[:, img_w:img_w * 2] = bridge.imgmsg_to_cv2(msg, "mono16")

### 1: x10 value -> to float -> rgb -> orb (numpy please)
### 2: x0.1 value -> gray uint 8 -> orb
### Story of calibration

for i in range(int(s_img0)):
    st = Time(ros_t[i].secs, ros_t[i].nsecs-1e6)
    if i > 0: st = Time(ros_t[i - 1].secs, ros_t[i - 1].nsecs + 1e6)
    et = Time(ros_t[i].secs, ros_t[i].nsecs)

    for topic, msg, t in bag.read_messages(topics=ros_msg, start_time=st, end_time=et):
        if topic == ros_msg[0]:
            img_prv[:, :img_w] = bridge.imgmsg_to_cv2(msg, "mono16")
        if topic == ros_msg[1]:
            img_prv[:, img_w:img_w*2] = bridge.imgmsg_to_cv2(msg, "mono16")
        # if topic == ros_msg[2]:
        #     print(msg.linear_acceleration)
        #     print(msg.angular_velocity)
            ...

    cv.imshow("***", img_prv)
    k = cv.waitKey(10)
    if k == 27:
        cv.destroyAllWindows()
        break
