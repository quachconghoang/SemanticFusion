import os.path
from os import path
import glob
import numpy as np
import open3d as o3d
import cv2 as cv

from Semantics.image_proc_2D import matching, dnn_device, getSuperPoints_v2, getAnchorPoints

from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

import sys
sys.path.append('../')

img_path = '/home/hoangqc/Datasets/AGZ/MAV_image_gt/'
depth_path = '/home/hoangqc/Datasets/AGZ/MAV_depth_npy/'
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

# Draw keypoints:
def draw_keypoints(img, kp, color=(0, 255, 0)):
    for i in range(len(kp)):
        cv.drawMarker(img, (int(kp[i, 0]), int(kp[i, 1])), color, cv.MARKER_CROSS, 10, 2)
    return img

# Get data from CSV file:
def get_data_from_csv(filename):
    data = np.genfromtxt(filename, delimiter=',')
    return data

imgs = get_image_files(img_path)
depths = get_image_files(depth_path, ext='npy')
img_distCoeff = np.load('/home/hoangqc/Datasets/AGZ/calibration_data/distCoeff.npy')[0]
img_intrinsic = np.load('/home/hoangqc/Datasets/AGZ/calibration_data/intrinsic_matrix.npy')
def_intrinsics = np.zeros((3, 4))
def_intrinsics[:3, :3] = img_intrinsic

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

# camera_vis = []
# for pose in transforms:
#     camFrame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
#     camFrame.transform(pose)
#     camera_vis.append(camFrame)

# camera_vis.extend([pcd_gps, pcd_xyz, axis_pcd])
# o3d.visualization.draw_geometries(camera_vis)

src = cv.imread(imgs[0], cv.IMREAD_UNCHANGED)
src_depth = np.load(depths[0])
tar = cv.imread(imgs[1], cv.IMREAD_UNCHANGED)
tar_depth = np.load(depths[1])

src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
tar_gray = cv.cvtColor(tar, cv.COLOR_RGB2GRAY)

pts0 = getSuperPoints_v2(src_gray)
pts1 = getSuperPoints_v2(tar_gray)

kp0 = pts0['pts']
kp1 = pts1['pts']
desc0 = pts0['desc']
desc1 = pts1['desc']

vis_src = draw_keypoints(src, kp0, color=(0, 255, 0))
vis_tar = draw_keypoints(tar, kp1, color=(255, 0, 0))

plt.rcParams['figure.dpi'] = 400
plt.imshow(vis_src)
plt.show()
plt.imshow(vis_tar)
plt.show()


kp0_uv = cv.undistortPoints(kp0, cameraMatrix=img_intrinsic, distCoeffs=img_distCoeff, P=def_intrinsics)
kp0_uv = kp0_uv.squeeze()

# save numpy to binary file
np.save('kp0.npy', kp0)