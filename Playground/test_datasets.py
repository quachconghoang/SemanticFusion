import rosbag
import datetime
import cv2

path = "/home/hoangqc/Datasets/TUM_VI/dataset-corridor1_512_16.bag"
ros_msg = ['/cam0/image_raw', '/cam1/image_raw', '/imu0', '/vrpn_client/raw_transform']

bag = rosbag.Bag(path)

t_start = bag.get_start_time()
count = 0;
for topic, msg, t in bag.read_messages(topics=ros_msg):
    if topic == ros_msg[0]:
        t_index = t.to_sec()-t_start;
        if t_index > count:
            print(t_index)
            count+=5

        if t_index > 30:
            break