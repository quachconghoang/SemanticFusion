import os.path
from os import path
import glob
import numpy as np
import open3d as o3d
import cv2 as cv

from scipy.spatial.transform import Rotation as R


from SlamUtils.visualization import getVisualizationBB, getKeyframe


import sys
sys.path.append('../')

img_path = '/home/hoangqc/Datasets/AGZ/MAV_image_gt/'
depth_path = '/home/hoangqc/Datasets/AGZ/MAV_depth/'
mav_gt_file = '/home/hoangqc/Datasets/AGZ/Log Files/GroundTruthAGL.csv'
# GroundTruthAGL.csv: imgid,
# x_gt, y_gt, z_gt, Yaw, Pitch, Roll,
# x_gps, y_gps, z_gps

def get_transforms_mat44(xyz, wpk):
    # Because the MAV's coordinate system is different from the camera's (From Z up to Y up, and same X axis)
    rotations = R.from_euler('xzy', wpk, degrees=True).as_matrix()
    transforms_mat44 = np.zeros((len(rotations), 4, 4))
    for i in range(len(rotations)):
        t = np.eye(4)
        t[:3, 3] = xyz[i]
        t[:3, :3] =  rotations[i]
        transforms_mat44[i] = t
    return transforms_mat44


# Get image file name from folder:
def get_image_files(folder, ext='jpg'):
    files = glob.glob(folder + '*.' +ext)
    files.sort()
    return files

# Get data from CSV file:
def get_data_from_csv(filename):
    data = np.genfromtxt(filename, delimiter=',')
    return data

imgs = get_image_files(img_path)
depths = get_image_files(depth_path, ext='png')
img_distCoeff = np.load('/home/hoangqc/Datasets/AGZ/calibration_data/distCoeff.npy')
img_intrinsic = np.load('/home/hoangqc/Datasets/AGZ/calibration_data/intrinsic_matrix.npy')
mav_gt = get_data_from_csv(mav_gt_file)[:-2, :10]
# mav_gt = get_data_from_csv(mav_gt_file)[:300, :10]

# Transform to MAV coordinate offset for visualization:
mav_gt = mav_gt - np.asarray([ 0,   mav_gt[0, 1], mav_gt[0, 2], mav_gt[0, 3],
                                    0, 0, 0,
                                    mav_gt[0, 7], mav_gt[0, 8], mav_gt[0, 9]])

xyz = mav_gt[:, 1:4]
ypr = mav_gt[:, 4:7]
transforms = get_transforms_mat44(xyz, ypr)

gps = mav_gt[:, 7:10]

# Set to zero for easy visualization:
# xyz = xyz-xyz[0]
# gps = gps-gps[0]

# Update location

pcd_xyz = o3d.geometry.PointCloud()
pcd_xyz.points = o3d.utility.Vector3dVector(xyz)
pcd_xyz.colors = o3d.utility.Vector3dVector(np.full(xyz.shape, [0, 255, 0]))

pcd_gps = o3d.geometry.PointCloud()
pcd_gps.points = o3d.utility.Vector3dVector(gps)
pcd_gps.colors = o3d.utility.Vector3dVector(np.full(gps.shape, [255, 0, 0]))

axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])

# Open3D gen Arrow from Poses:
def generateArrowFromPoses(poses_mat44):
    arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.05, cone_radius=0.08, cylinder_height=0.5, cone_height=0.2)
    # arrow.compute_vertex_normals()
    arrow.transform(poses_mat44)
    return arrow

camera_vis = []
for pose in transforms:
    camFrame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    camFrame.transform(pose)
    camera_vis.append(camFrame)
    # getVisualizationBB(pcd_xyz, camFrame)

camera_vis.extend([pcd_gps, pcd_xyz, axis_pcd])
o3d.visualization.draw_geometries(camera_vis)

# for i in range(len(mav_gt)):
#     img = cv.imread(imgs[i], cv.IMREAD_UNCHANGED)
#     cv.imshow('image', img)
#     cv.waitKey(10)
#
# cv.destroyAllWindows()