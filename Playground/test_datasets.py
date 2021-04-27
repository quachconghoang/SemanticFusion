import rosbag
import datetime
import cv2 as cv
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from queue import Queue

VIODE_msg = ['/cam0/image_raw', '/cam0/segmentation', '/cam1/image_raw', '/cam1/segmentation', '/imu0', '/odometry']
TUMVI_msg = ['/cam0/image_raw', '/cam1/image_raw', '/imu0', '/vrpn_client/raw_transform']

path = "/home/hoangqc/Datasets/TUM_VI/dataset-corridor1_512_16.bag"
path_VIODE = "/home/hoangqc/Datasets/VIODE/city_day/0_none.bag"

ros_msg = TUMVI_msg
bridge = CvBridge()

bag = rosbag.Bag(path)
img0 = img1 = np.zeros([512,512],dtype=np.int16)

t_start = bag.get_start_time()
s_img0 = bag.get_message_count(topic_filters=[ros_msg[0]])
s_img1 = bag.get_message_count(topic_filters=[ros_msg[1]])

ros_t = []

for topic, msg, t in bag.read_messages(topics=[ros_msg[0]]):
    ros_t.append(t)
    # print(t)
    ...

gt = []
for i in range(int(s_img0-1)):
    for topic, msg, t in bag.read_messages(topics=ros_msg, start_time=ros_t[i], end_time=ros_t[i+1]):
        if topic == ros_msg[0]:
            img0 = bridge.imgmsg_to_cv2(msg, "mono16")
        if topic == ros_msg[1]:
            img1 = bridge.imgmsg_to_cv2(msg, "mono16")

    cv.imshow("0", img0)
    cv.imshow("1", img1)
    if cv.waitKey() == 27: break

# for topic, msg, t in bag.read_messages(topics=ros_msg[3], start_time=msg_times[0],end_time=msg_times[1]):
    # gt.append(msg)
    # print(msg)

# for topic, msg, t in bag.read_messages(topics=ros_msg):
#     t_index = t.to_sec() - t_start;
#
#     if topic == ros_msg[0]:
#         t_index = t.to_sec()-t_start;

    #     if t_index > count:
    #         print(msg.encoding)
    #         # break
    #         imgMsg = bridge.imgmsg_to_cv2(msg, "mono8") #bgr8 bgr16 rgb8 rgb16 mono8 mono16
    #         cv.imshow("zzz",imgMsg)
    #         k = cv.waitKey()
    #         if k == 27:
    #             break
            # count +=1

        # if t_index > 30:
        #     break