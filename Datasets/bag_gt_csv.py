import rospy
import rosbag
import pandas as pd

rospy.init_node('reader', anonymous=True)
rootDir = 'C:/Users/hoangqc/Desktop/Datasets/'
bag = rosbag.Bag(rootDir + 'Airsim-ros/FP-loop0-752x480.bag')

topic_info = bag.get_type_and_topic_info()

message = '/groundtruth/pose'
# message = '/odometry'

useNED=True

column_names = ['#timestamp', 'p_RS_R_x [m]', 'p_RS_R_y [m]', 'p_RS_R_z [m]', 'q_RS_w []', 'q_RS_x []', 'q_RS_y []', 'q_RS_z []',
                '#timestamp(s)', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']

pose0 = [0,0,0]
load_offset = False

df = pd.DataFrame(columns=column_names)

for topic, msg, t in bag.read_messages(topics=[message]):
    msg_pose = msg.pose
    # msg_pose = msg.pose.pose

    if load_offset == False:
        load_offset = True
        pose0 = [msg_pose.position.x,
                 msg_pose.position.y,
                 msg_pose.position.z]

    time_ns = msg.header.stamp

    pose = [(msg_pose.position.x - pose0[0]),
            (msg_pose.position.y - pose0[1]),
            (msg_pose.position.z - pose0[2])]

    orient = [msg_pose.orientation.w,
            msg_pose.orientation.x,
            msg_pose.orientation.y,
            msg_pose.orientation.z]

    if(useNED):
        pose = [(msg_pose.position.x - pose0[0]),
                -(msg_pose.position.y - pose0[1]),
                -(msg_pose.position.z - pose0[2])]

        orient = [msg_pose.orientation.w,
                  -msg_pose.orientation.x,
                  -msg_pose.orientation.y,
                  msg_pose.orientation.z]

    df = df.append(
        {column_names[0] : msg.header.stamp,
         column_names[1] : pose[0],
         column_names[2] : pose[1],
         column_names[3] : pose[2],
         column_names[4]: orient[0],
         column_names[5]: orient[1],
         column_names[6]: orient[2],
         column_names[7]: orient[3],

         column_names[8]: msg.header.stamp.to_sec(),
         column_names[9]: pose[0],
         column_names[10]: pose[1],
         column_names[11]: pose[2],
         column_names[12]: orient[1],
         column_names[13]: orient[2],
         column_names[14]: orient[3],
         column_names[15]: orient[0]
         },
        ignore_index=True
    )

    # print(msg.header.stamp, pose, orient)

outName = 'FP-loop0-752x480'

if useNED:
    df.to_csv(rootDir + outName + '-vis.csv')
else:
    df.to_csv(rootDir + outName + '.csv')

# df.to_csv(rootDir + 'FP-loop0-752x480.csv')