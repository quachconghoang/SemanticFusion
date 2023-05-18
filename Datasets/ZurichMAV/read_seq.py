import os.path
from os import path
import glob
import numpy as np
import open3d as o3d
import cv2 as cv

from scipy.spatial.transform import Rotation as R

import sys
sys.path.append('../')

img_path = '/home/hoangqc/Datasets/AGZ/MAV_Images/'
depth_path = '/home/hoangqc/Datasets/AGZ/MAV_Depths/'
mav_gt_file = '/home/hoangqc/Datasets/AGZ/Log Files/GroundTruthAGL.csv'
# GroundTruthAGL.csv: imgid,
# x_gt, y_gt, z_gt, Yaw, Pitch, Roll,
# x_gps, y_gps, z_gps

# Get Quaternions from Euler angles by Scipy:
def get_quaternions_from_euler(rpy):
    r = R.from_euler('xyz', rpy, degrees=True)
    q = r.as_quat()
    return q

# Get image file name from folder:
def get_image_files(folder):
    files = glob.glob(folder + '*.jpg')
    files.sort()
    return files

# Get data from CSV file:
def get_data_from_csv(filename):
    data = np.genfromtxt(filename, delimiter=',')
    return data

# Read images and imshow by opencv:
def read_images(imgs):
    for img in imgs:
        img = cv.imread(img)
        cv.imshow('image', img)
        cv.waitKey(30)
        # cv.destroyAllWindows()

imgs = get_image_files(img_path)
img_distCoeff = np.load('/home/hoangqc/Datasets/AGZ/calibration_data/distCoeff.npy')
img_intrinsic = np.load('/home/hoangqc/Datasets/AGZ/calibration_data/intrinsic_matrix.npy')
mav_gt = get_data_from_csv(mav_gt_file)[:-2, :10]

# 15:18
img_ids = (mav_gt[:, 0]-1).astype(int)
imgs = np.take(imgs, img_ids)

xyz = mav_gt[:, 1:4]
rpy = mav_gt[:, 4:7]
gps = mav_gt[:, 7:10]

# Set to zero for easy visualization:
xyz = xyz-xyz[0]
gps = gps-gps[0]

pcd_xyz = o3d.geometry.PointCloud()
pcd_xyz.points = o3d.utility.Vector3dVector(xyz)
pcd_xyz.colors = o3d.utility.Vector3dVector(np.full(xyz.shape, [0, 255, 0]))

pcd_gps = o3d.geometry.PointCloud()
pcd_gps.points = o3d.utility.Vector3dVector(gps)
pcd_gps.colors = o3d.utility.Vector3dVector(np.full(gps.shape, [255, 0, 0]))

axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=30.0, origin=[0, 0, 0])
o3d.visualization.draw_geometries([pcd_gps, pcd_xyz, axis_pcd])

# read_images(imgs)
# cv.destroyAllWindows()

### xyz poses to motions:
# def poses_to_motions(poses):
#     motions = []
#     for i in range(poses.shape[0]-1):
#         motion = poses[i+1] - poses[i]
#         motions.append(motion)
#     return np.array(motions)
#
# motions = poses_to_motions(xyz) # Notes: average speed is 0.7m/s
