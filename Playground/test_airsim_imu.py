import airsim
from pathlib import Path
import numpy as np

import rospy
from rospy import Time
import rosbag
from sensor_msgs.msg import Imu, Image, CompressedImage
from geometry_msgs.msg import Pose, PoseStamped, PoseArray, Quaternion
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError


_compress = False
# saveDir = str(Path.home())+ '/Datasets/Airsim/'
client = airsim.MultirotorClient()
client.confirmConnection()

rospy.init_node('writer', anonymous=True)
bridge = CvBridge()
bag = rosbag.Bag('C:/Users/hoangqc/Desktop/Datasets/FP-loop0-752x480.bag', 'w')

ctime = Time.now()
# ctime_sec = ctime.to_sec()

count_img = 0
count_IMU = 0
interval_IMU = 0.005
interval_img = 0.05

def getDroneArmed():
    thrust = client.getRotorStates().rotors[0]['thrust']
    return thrust > 0.1

def waitForTakeOff():
    # is_flying = getDroneArmed()
    # while is_flying==False:
    #     is_flying = getDroneArmed()
    is_flying = client.getMultirotorState().landed_state
    while is_flying==0:
        is_flying = client.getMultirotorState().landed_state
    print("taking off !!!!!!!!!!!!!!!!!!")
    print("armed!!!")
    # print("taking off !!!!!!!!!!!!!!!!!!")
    # client.simContinueForTime()


def getIMU_message(imu_data, pose_gt, count = 0, time=Time.now()):
    m_Imu = Imu()
    m_Imu.header.seq = count
    m_Imu.header.stamp.set(time.secs, time.nsecs)
    m_Imu.orientation.w = imu_data.orientation.w_val
    m_Imu.orientation.x = imu_data.orientation.x_val
    m_Imu.orientation.y = imu_data.orientation.y_val
    m_Imu.orientation.z = imu_data.orientation.z_val

#insert S.I.M here !!!
    m_Imu.angular_velocity.x = imu_data.angular_velocity.x_val
    m_Imu.angular_velocity.y = imu_data.angular_velocity.y_val
    m_Imu.angular_velocity.z = imu_data.angular_velocity.z_val
    m_Imu.linear_acceleration.x = imu_data.linear_acceleration.x_val
    m_Imu.linear_acceleration.y = imu_data.linear_acceleration.y_val
    m_Imu.linear_acceleration.z = imu_data.linear_acceleration.z_val

    # m_Odom = Odometry()
    # m_Odom.header.seq = count
    # m_Odom.header.stamp.set(time.secs, time.nsecs)
    # m_Odom.pose.pose.position.x = pose_gt.position.x_val
    # m_Odom.pose.pose.position.y = pose_gt.position.y_val
    # m_Odom.pose.pose.position.z = pose_gt.position.z_val
    # m_Odom.pose.pose.orientation.w = pose_gt.orientation.w_val
    # m_Odom.pose.pose.orientation.x = pose_gt.orientation.x_val
    # m_Odom.pose.pose.orientation.y = pose_gt.orientation.y_val
    # m_Odom.pose.pose.orientation.z = pose_gt.orientation.z_val

    m_Pose = PoseStamped()
    m_Pose.header.seq = count
    m_Pose.header.stamp.set(time.secs, time.nsecs)
    m_Pose.pose.position.x = pose_gt.position.x_val
    m_Pose.pose.position.y = pose_gt.position.y_val
    m_Pose.pose.position.z = pose_gt.position.z_val
    m_Pose.pose.orientation.w = pose_gt.orientation.w_val
    m_Pose.pose.orientation.x = pose_gt.orientation.x_val
    m_Pose.pose.orientation.y = pose_gt.orientation.y_val
    m_Pose.pose.orientation.z = pose_gt.orientation.z_val


    return m_Imu, m_Pose


def getBinocularImg(count = 0, time=Time.now(), need_compress = False):
    responds = client.simGetImages([
        airsim.ImageRequest("RGB_Left", airsim.ImageType.Scene, compress=need_compress),
        airsim.ImageRequest("RGB_Right", airsim.ImageType.Scene, compress=need_compress)
    ])
    # get numpy array
    imgL1d = np.frombuffer(responds[0].image_data_uint8, dtype=np.uint8)
    imgL_rgb = imgL1d.reshape(responds[0].height, responds[0].width, 3)

    imgR1d = np.frombuffer(responds[1].image_data_uint8, dtype=np.uint8)
    imgR_rgb = imgR1d.reshape(responds[1].height, responds[1].width, 3)

    msg_L = bridge.cv2_to_imgmsg(cvim=imgL_rgb, encoding="bgr8")
    msg_L.header.seq = count
    msg_L.header.stamp.set(time.secs, time.nsecs)

    msg_R = bridge.cv2_to_imgmsg(cvim=imgR_rgb, encoding="bgr8")
    msg_R.header.seq = count
    msg_R.header.stamp.set(time.secs, time.nsecs)

    return msg_L, msg_R

def getGroundTruthImg(count = 0, time=Time.now(), need_compress = False):
    responds = client.simGetImages([
        airsim.ImageRequest("RGB_Left", airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=need_compress),
        airsim.ImageRequest("RGB_Left", airsim.ImageType.Segmentation, compress=need_compress)
    ])
    # get numpy array
    img_depth = airsim.list_to_2d_float_array(responds[0].image_data_float, responds[0].width, responds[0].height)

    img_buf = np.frombuffer(responds[1].image_data_uint8, dtype=np.uint8)
    imgR_seg = img_buf.reshape(responds[1].height, responds[1].width, 3)

    msg_depth = bridge.cv2_to_imgmsg(cvim=img_depth, encoding="passthrough")
    msg_depth.header.seq = count
    msg_depth.header.stamp.set(time.secs, time.nsecs)

    msg_seg = bridge.cv2_to_imgmsg(cvim=imgR_seg, encoding="bgr8")
    msg_seg.header.seq = count
    msg_seg.header.stamp.set(time.secs, time.nsecs)

    return msg_depth, msg_seg

if __name__ == "__main__":
    waitForTakeOff()
    is_flying = client.getMultirotorState().landed_state

    while is_flying:
        if count_IMU == 0:
            ctime = Time.now()

        if client.simIsPause() == False:
            client.simPause(True)

        if count_IMU % 5 == 0:
            img_Time = Time.from_sec(ctime.to_sec() + count_img * interval_img)
            msg_L, msg_R = getBinocularImg(count=count_img, time=img_Time, need_compress=False)

            bag.write(topic='/cam0/image_raw', msg=msg_L, t=img_Time)
            bag.write(topic='/cam1/image_raw', msg=msg_R, t=img_Time)
            print('- Get Image: ', count_img)
            count_img += 1

        # if count_IMU % 100 == 0:
        #     msg_depth, msg_seg = getGroundTruthImg(count=count_img, time=img_Time, need_compress=False)
        #     bag.write(topic='/cam0/depth', msg=msg_depth, t=img_Time)
        #     bag.write(topic='/cam0/segmentation', msg=msg_seg, t=img_Time)
        #     print('- - - Get GroundTruth: ', count_img)
        #     ...

        for i in range(10):
            imu_Time = Time.from_sec(ctime.to_sec() + count_IMU * interval_IMU)
            imu_data = client.getImuData(imu_name="IMU", vehicle_name="CVFlight")
            pose_gt = client.simGetVehiclePose("CVFlight")

            imu_msg, pose_msg = getIMU_message(imu_data=imu_data, pose_gt=pose_gt, count=count_IMU, time=imu_Time)

            bag.write(topic='/imu0', msg=imu_msg, t=imu_Time)
            # bag.write(topic='/groundtruth/odometry', msg=odom_msg, t=imu_Time)
            if i==0:
                bag.write(topic='/groundtruth/pose', msg=pose_msg, t=imu_Time)
                print('GetIMU: ', count_IMU)

            count_IMU += 1
            client.simContinueForTime(seconds=interval_IMU)

        is_flying = client.getMultirotorState().landed_state

    bag.close()
    client.simPause(False)