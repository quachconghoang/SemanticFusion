import rospy
import rosbag
import pandas as pd

rospy.init_node('reader', anonymous=True)
bag = rosbag.Bag('/home/hoangqc/Datasets/Airsim-ros/NH-base-5ms.bag')

topic_info = bag.get_type_and_topic_info()

message = '/groundtruth/pose'

column_names = ['#timestamp [ns]', 'p_RS_R_x [m]', 'p_RS_R_y [m]', 'p_RS_R_z [m]', 'q_RS_w []', 'q_RS_x []', 'q_RS_y []', 'q_RS_z []']

pose0 = [0,0,0]
load_offset = False

df = pd.DataFrame(columns=column_names)

for topic, msg, t in bag.read_messages(topics=[message]):
    if load_offset == False:
        load_offset = True
        pose0 = [msg.pose.position.x,
                 msg.pose.position.y,
                 msg.pose.position.z]

    time_ns = msg.header.stamp
    pose = [(msg.pose.position.x - pose0[0]),
            -(msg.pose.position.y - pose0[1]),
            -(msg.pose.position.z - pose0[2])]

    orient = [msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z]
    df = df.append(
        {column_names[0] : msg.header.stamp,
         column_names[1] : pose[0],
         column_names[2] : pose[1],
         column_names[3] : pose[2],
         column_names[4]: orient[0],
         column_names[5]: orient[1],
         column_names[6]: orient[2],
         column_names[7]: orient[3],
         },
        ignore_index=True
    )

    # print(msg.header.stamp, pose, orient)

df.to_csv('~/NH-base-5ms.csv')