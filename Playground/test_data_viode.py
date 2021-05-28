import rosbag
from rospy import Time
import cv2 as cv
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from queue import Queue
import numpy as np
import pandas as pd
from pathlib import Path

VIODE_msg = ['/cam0/image_raw', '/cam0/segmentation',
             '/cam1/image_raw', '/cam1/segmentation',
             '/imu0',
             '/odometry']
VIODE_path = str(Path.home())+ "/Datasets/VIODE/city_day/1_low.bag"
saveDir = str(Path.home())+ '/Datasets/VIODE/'

ros_msg = VIODE_msg
bridge = CvBridge()

img_w = 752
img_h = 480
count = 0

bag = rosbag.Bag(VIODE_path)
# img0 = img1 = np.zeros([512,512], dtype=np.uint16)
# img_prv = np.zeros([img_h*2,img_w*2, 3], dtype=np.uint8)
img_prv = np.zeros([img_h,img_w*2, 3], dtype=np.uint8)
img = np.zeros([img_h,img_w, 3], dtype=np.uint8)

t_start = bag.get_start_time()
s_img0 = bag.get_message_count(topic_filters=[ros_msg[0]])
s_img1 = bag.get_message_count(topic_filters=[ros_msg[2]])

ros_t = []
ros_t.clear()
for topic, msg, t in bag.read_messages(topics=[ros_msg[0]]):
    ros_t.append(t)

# pd.DataFrame(np.array(ros_t)).to_csv("viode_timestamp.csv")

for i in range(int(s_img0)):
    st = Time(ros_t[i].secs, ros_t[i].nsecs-1e6)
    if i > 0: st = Time(ros_t[i - 1].secs, ros_t[i - 1].nsecs + 1e6)
    et = Time(ros_t[i].secs, ros_t[i].nsecs)
    # print(et.to_nsec() - st.to_nsec())

    for topic, msg, t in bag.read_messages(topics=ros_msg, start_time=st, end_time=et):
        if topic == ros_msg[0]:
            img = bridge.imgmsg_to_cv2(msg, "bgr8")

        # if topic == ros_msg[1]:
        #     img_prv[img_h:, :img_w, :] = bridge.imgmsg_to_cv2(msg, "bgr8")
            ...

        # if topic == ros_msg[2]:
            # img_prv[:img_h, img_w:, :] = bridge.imgmsg_to_cv2(msg, "bgr8")
        # if topic == ros_msg[3]:
        #     img_prv[img_h:, img_w:, :] = bridge.imgmsg_to_cv2(msg, "bgr8")
            ...

        # if topic == '/imu0':
        #     print(topic,msg)
        #
        # if topic == '/odometry':
        #     print(topic,msg)

    cv.imshow("|*| --- *** --- |*|", img)
    k = cv.waitKey()
    if k == 27:
        cv.destroyAllWindows()
        break


    if k == ord('s'):
        name = saveDir + str(count).zfill(5) + '.png'
        cv.imwrite(filename=name, img=img)
        count = count+1