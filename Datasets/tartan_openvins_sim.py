import glob, os
from pathlib import Path
import cv2 as cv
import sys
sys.path.append('../')
import numpy as np
import datetime

from SlamUtils.Loader.TartanAir import getRootDir, getDataSequences, getDataLists
rootDIR = getRootDir()

def convertBags(_pathTartan, _pathBag, _bagName = '0x.bag'):

    import rospy, rosbag
    from rospy import Time
    from sensor_msgs.msg import Imu, Image, CompressedImage
    from geometry_msgs.msg import Pose, PoseStamped, PoseArray, Quaternion
    from nav_msgs.msg import Odometry
    from cv_bridge import CvBridge, CvBridgeError

    files_rgb_left, files_rgb_right, files_depth_left, poselist = getDataLists(dir=_pathTartan, skip=1)

    imulist_t = np.loadtxt(_pathBag + 'imu.txt')
    poselist_t = np.loadtxt(_pathBag + 'pose_gt.txt')

    # if have offset
    data_ids = np.loadtxt(_pathBag + 'data_id.txt').astype(np.int32).tolist()
    files_rgb_left = [files_rgb_left[i] for i in data_ids]
    files_rgb_right = [files_rgb_right[i] for i in data_ids]

    # bag_name = 'office_' + 'easy_' + '004' + '.bag'
    rospy.init_node('writer', anonymous=True)
    bridge = CvBridge()
    bag = rosbag.Bag(_pathBag + _bagName, 'w')

    st = datetime.datetime(year=2023, month=3, day=23, hour=23, minute=23, second=23)
    for i in range(imulist_t.shape[0]):
        # cTime = (st + datetime.timedelta(milliseconds=(i * delta_imu))).timestamp()
        imu = imulist_t[i]
        cTime = rospy.Time.from_seconds(imu[0])

        m_Imu = Imu()
        m_Imu.header.seq = i
        m_Imu.header.stamp = cTime

        m_Imu.angular_velocity.x = imu[1]
        m_Imu.angular_velocity.y = imu[2]
        m_Imu.angular_velocity.z = imu[3]

        m_Imu.linear_acceleration.x = imu[4]
        m_Imu.linear_acceleration.y = imu[5]
        m_Imu.linear_acceleration.z = imu[6]

        bag.write(topic='/imu0', msg=m_Imu, t=cTime)

    for i in range(poselist_t.shape[0]):
        pose = poselist_t[i]
        print('img ', i, ' ...')

        m_Pose = PoseStamped()
        cTime = rospy.Time.from_seconds(pose[0])

        m_Pose.header.seq = i
        m_Pose.header.stamp = cTime
        m_Pose.pose.position.x = pose[1]
        m_Pose.pose.position.y = pose[2]
        m_Pose.pose.position.z = pose[3]
        m_Pose.pose.orientation.x = pose[4]
        m_Pose.pose.orientation.y = pose[5]
        m_Pose.pose.orientation.z = pose[6]
        m_Pose.pose.orientation.w = pose[7]

        bag.write(topic='/groundtruth/pose', msg=m_Pose, t=cTime)

        imgL = cv.imread(files_rgb_left[i])
        imgR = cv.imread(files_rgb_right[i])
        msg_L = bridge.cv2_to_imgmsg(cvim=imgL, encoding="bgr8")
        msg_L.header.seq = i
        msg_L.header.stamp = cTime
        msg_R = bridge.cv2_to_imgmsg(cvim=imgR, encoding="bgr8")
        msg_R.header.seq = i
        msg_R.header.stamp = cTime

        bag.write(topic='/cam0/image_raw', msg=msg_L, t=cTime)
        bag.write(topic='/cam1/image_raw', msg=msg_R, t=cTime)

    bag.close()

def convertPoseSequence(poses, offset = 0):
    start_time = datetime.datetime(year=2022, month=2, day=22, hour=22, minute=22, second=22)

    initSecs = 2
    fps = 20
    delta_ms = 50 # 1000 / fps

    data_count = poses.shape[0] - offset
    init_frames = fps * initSecs
    total_frames = init_frames + data_count

    state_id = []
    original_ids = [*range(offset, offset + data_count)]
    init_ids = [*range(offset,offset+20)]
    state_id.extend(init_ids)
    state_id.extend(init_ids[::-1])
    state_id.extend(original_ids)

    times = np.zeros(total_frames)
    for i in range(total_frames):
        print(i)
        times[i] = (start_time + datetime.timedelta(milliseconds=(i * delta_ms))).timestamp()

    times = times.reshape([-1, 1])

    new_poses = poses[state_id]
    poselist_wTime = np.concatenate((times, new_poses), axis=1)

    state_id = np.asarray(state_id).astype(np.int32).reshape([-1,1])
    # state_id_wTime = np.concatenate((times,state_id), axis=1)

    return poselist_wTime, state_id

if __name__ == "__main__":
    # fname = save_dir + 'pose_gt.txt'
    header = '# t x y z qx qy qz qw'
    seq_name = 'office'
    level = 'Easy'

    path = getDataSequences(root=rootDIR, scenario='office', level='Easy', seq_num=4)
    path_dirs = list(filter(None, path.split('/')))[::-1]
    save_dir = Path(str(Path.home()) + '/Datasets/TartanAir_Bag/' + seq_name + '/' + path_dirs[1] + '/' + path_dirs[0] + '/')
    save_dir.mkdir(parents=True, exist_ok=True)
    # if save_dir.exists():

    files_rgb_left, files_rgb_right, files_depth_left, poselist = getDataLists(dir=path, skip=1)

    # poses_mat34 = pos_quats2SEs(poselist)  # [R|t - array 12]
    # motions_mat = pose2motion(poses_mat34)  # [R|t]
    # motions_quat = SEs2ses(motions_mat).astype(np.float32)  # x-y-z qx-qy-qz-qw
    # trans_vec = motions_quat[:, :3]
    # rots_vec = motions_quat[:, 3:]
    # speeds = [np.linalg.norm(trans) * 20 for trans in trans_vec]

    poselist_wTime, data_ids = convertPoseSequence(poselist, offset=0)

    # poselist_wTime, data_ids = (poselist, offset=offset)

    np.savetxt(fname = save_dir._str + '/pose_gt.txt', X=poselist_wTime, header=header)
    np.savetxt(fname = save_dir._str + '/data_id.txt', X=data_ids, fmt='%i', header='data index')

    #   RUN OpenVINS

    #   ...
    # convertBags(path, save_dir, _bagName= seq_name+'.bag')
    convertBags(path, save_dir._str+'/')