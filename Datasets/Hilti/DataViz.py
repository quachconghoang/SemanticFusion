import rospy
import rosbag
from rospy import Time
from sensor_msgs.msg import Imu, Image, CompressedImage, LaserScan
from geometry_msgs.msg import Pose, PoseStamped, PoseArray, Quaternion, Vector3
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError

import open3d as o3d
import ros_numpy
import numpy as np
from scipy.spatial.transform import Rotation as R

bag_file = '/home/hoangqc/Datasets/Hilti-2022/Additional Sequences/exp04_construction_upper_level.bag'
# bag_file = '/home/hoangqc/Datasets/Hilti-2022/Additional Sequences/exp05_construction_upper_level_2.bag'
# bag_file = '/home/hoangqc/Datasets/Hilti-2022/Additional Sequences/exp06_construction_upper_level_3.bag'

gt_file = '/home/hoangqc/Datasets/Hilti-2022/construction_site_upper_level_1cm.ply'
pcd_gt = o3d.io.read_point_cloud(gt_file)
voxel_down_pcd = pcd_gt.voxel_down_sample(voxel_size=0.05)

delta_t = 0.005
t_lidar = np.loadtxt('./Datasets/Hilti/exp04_time_lidar.txt')
bag_file_gt = '/home/hoangqc/Datasets/Hilti-2022/Additional Sequences/exp_04_construction_upper_level_imu.txt'

def get_transforms_mat44(gt_liegroups):
    xyz = gt_liegroups[:, :3]
    quads = gt_liegroups[:, 3:]
    rotations = R.from_quat(quads).as_matrix()
    transforms_mat44 = np.zeros((len(rotations), 4, 4))
    for i in range(len(rotations)):
        t = np.eye(4)
        t[:3, 3] = xyz[i]
        t[:3, :3] =  rotations[i]
        transforms_mat44[i] = t
    return transforms_mat44

poses_gt = np.loadtxt(bag_file_gt)
gt_timestamps = poses_gt[:, 0]
gt_liegroups = poses_gt[:, 1:]
gt_mat44 = get_transforms_mat44(gt_liegroups)
lidar_extrinsics = [-0.001, -0.00855, 0.055, 0.7071068, -0.7071068, 0, 0]
lidar_extrinsics = np.array(lidar_extrinsics).reshape((1, 7))
lidar_extrinsics_mat44 = get_transforms_mat44(lidar_extrinsics).squeeze()

# https://github.com/sevensense-robotics/core_research_manual/blob/master/pages/getting_started.md
msgs ={
    'imu': '/alphasense/imu', # 400Hz
    'cam0': '/alphasense/cam0/image_raw', # 40Hz
    'cam1': '/alphasense/cam1/image_raw',
    'cam2': '/alphasense/cam2/image_raw',
    'cam3': '/alphasense/cam3/image_raw',
    'cam4': '/alphasense/cam4/image_raw',
    'lidar': '/hesai/pandar', # 10Hz
}

rospy.init_node('reader', anonymous=True)
bridge = CvBridge()
bag = rosbag.Bag(bag_file)

def getLidar(tSec, dt = 0.005):
    for topic, msg, t in bag.read_messages(start_time=Time.from_sec(tSec-dt), end_time=Time.from_sec(tSec+dt) , topics = msgs['lidar']):
        return msg

def getCameras(tSec, dt = 0.005):
    for topic, msg, t in bag.read_messages(start_time=Time.from_sec(tSec-dt), end_time=Time.from_sec(tSec+dt) , topics = msgs['cam0']):
        return msg

start_id = 0
msg_lidar = getLidar(tSec=t_lidar[start_id])
points = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg_lidar, remove_nans=False)
# Open3D display Point Cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.transform(np.matmul(gt_mat44[start_id], lidar_extrinsics_mat44))
# axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
# o3d.visualization.draw_geometries([axis_pcd, pcd, voxel_down_pcd])


#3D VISUALIZATION
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name='Trajectory', width=1280, height=768)
# bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-15, -15, -5), max_bound=(15, 15, 5))

axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=[0, 0, 0])
cam_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
vis.add_geometry(axis_pcd)
vis.add_geometry(voxel_down_pcd)
vis.add_geometry(pcd)

viewControl = vis.get_view_control()
vis.poll_events();vis.update_renderer()

for id in range(t_lidar.shape[0]):
    msg_lidar = getLidar(tSec=t_lidar[id])
    points = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg_lidar, remove_nans=False)
    pcd.points = o3d.utility.Vector3dVector(points)
    real_trans = np.matmul(gt_mat44[id], lidar_extrinsics_mat44)
    pcd.transform( real_trans)

    # cam_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    cam_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    cam_pcd.transform(real_trans)
    vis.add_geometry(cam_pcd)

    # vis.get_view_control().look_at(gt_liegroups[id][:3])
    vis.update_geometry(pcd)
    viewControl.set_lookat(gt_liegroups[id][:3])

    vis.poll_events()
    vis.update_renderer()


vis.poll_events();vis.update_renderer();vis.run()
vis.destroy_window()