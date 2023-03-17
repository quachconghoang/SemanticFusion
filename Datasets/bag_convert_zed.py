import rosbag
import rospy

from rospy import Time
import datetime
import cv2 as cv
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import pandas as pd
from enum import Enum

rospy.init_node('reader', anonymous=True)
bag = rosbag.Bag('C:/Users/hoangqc/Desktop/Datasets/datatest3.bag')
bag_tuned = rosbag.Bag('C:/Users/hoangqc/Desktop/Datasets/datatest3_mode.bag', 'w')

topic_info = bag.get_type_and_topic_info()

message_imu = '/ZED_2/zed2/zed_node/imu/data_raw'
message_img = '/ZED_2/zed2/zed_node/stereo_raw/image_raw_color'
bag_timestamp=[]
img_count = bag.get_message_count(topic_filters=[message_img])
for topic, msg, t in bag.read_messages(topics=[message_img]):
    bag_timestamp.append(t)

bridge = CvBridge()

for topic, msg, t in bag.read_messages(topics=[message_imu, message_img]):
    if topic == message_img:
        print(t)
        img_raw = bridge.imgmsg_to_cv2(msg, "passthrough")
        img_rgb = img_raw[:, :, 0:3]
        img_l = img_rgb[:,:672:,:]
        img_r = img_rgb[:,672:1344,:]

        msg_l = bridge.cv2_to_imgmsg(cvim=img_l, encoding="bgr8")
        msg_l.header.stamp = t
        bag_tuned.write(topic='/cam0/image_raw', msg=msg_l, t=t)

        msg_r = bridge.cv2_to_imgmsg(cvim=img_r, encoding="bgr8")
        msg_r.header.stamp = t
        bag_tuned.write(topic='/cam1/image_raw', msg=msg_r, t=t)


        ...
        # self.img_0 = self.bridge.imgmsg_to_cv2(msg, self.img_type)
        # self.img_prv[:, :self.img_w] = self.bridge.imgmsg_to_cv2(msg, self.img_type)
    if topic == message_imu:
        bag_tuned.write(topic='/imu0', msg=msg, t=t)
        ...
        # self.img_1 = self.bridge.imgmsg_to_cv2(msg, self.img_type)
        # self.img_prv[:, self.img_w:(self.img_w * 2)] = self.bridge.imgmsg_to_cv2(msg, self.img_type)

    if topic == '/odometry':
        odom = msg

bag_tuned.close()
# def getImg(topic_name="topic_name", index=0):
#     i = index
#     st = Time(bag_timestamp[i].secs, bag_timestamp[i].nsecs - 1e6)
#     if i > 0: st = Time(bag_timestamp[i - 1].secs, bag_timestamp[i - 1].nsecs + 1e6)
#     et = Time(bag_timestamp[i].secs, bag_timestamp[i].nsecs)
#     for topic, msg, t in bag.read_messages(topics=[topic_name], start_time=st, end_time=et):
#         img_0 = bridge.imgmsg_to_cv2(msg, "passthrough")
#         return img_0
#
# for i in range(len(bag_timestamp)):
#     print(i)
#     img = getImg(topic_name=message_img, index=i)
#     cv.imshow("wtf", img)
#     cv.waitKey(100)
#     # cv.destroyAllWindows()
# cv.destroyAllWindows()

# bag_tuned.write(topic='/groundtruth/pose', msg=pose_msg, t=imu_Time)
# bag_tuned.write(topic='/imu0', msg=imu_msg, t=imu_Time)
# bag_tuned.write(topic='/cam0/image_raw', msg=msg_L, t=img_Time)
# bag_tuned.write(topic='/cam1/image_raw', msg=msg_R, t=img_Time)
