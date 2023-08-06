import os.path
from os import path
import glob
import numpy as np
import open3d as o3d
import cv2 as cv
from scipy.spatial.transform import Rotation as R
import sys
sys.path.append('../')

img_path = '/home/hoangqc/Datasets/AGZ/MAV_image_gt/'
depth_path = '/home/hoangqc/Datasets/AGZ/MAV_depth_npy/'
mav_gt_file = '/home/hoangqc/Datasets/AGZ/Log Files/GroundTruthAGL.csv'

# GroundTruthAGL.csv: imgid: x, y, z, omega(X), phi(Y), kappa(Z)
# - Kappa (κ), the rotation around the Z axis.
# - Phi (φ), the rotation around the Y axis.
# - Omega (ω), the rotation around the Χ axis
def get_transforms_mat44(xyz, wpk):
    rotations = R.from_euler('zyx', wpk, degrees=True).as_matrix()
    transforms_mat44 = np.zeros((len(rotations), 4, 4))
    for i in range(len(rotations)):
        t = np.eye(4)
        t[:3, 3] = xyz[i]
        t[:3, :3] =  rotations[i]
        transforms_mat44[i] = t
    return transforms_mat44


def get_transforms_quads_colmap(xyz, wpk):
    rotations = R.from_euler('zyx', wpk, degrees=True).as_quat()
    transforms_quads = np.zeros((len(rotations), 7))
    # convert to qw, qx, qy, qz, tx, ty, tz
    for i in range(len(rotations)):
        transforms_quads[i][4:] = xyz[i]
        transforms_quads[i][:4] = rotations[i]#[[3, 0, 1, 2]]
    return transforms_quads

img_distCoeff = np.load('/home/hoangqc/Datasets/AGZ/calibration_data/distCoeff.npy')[0]
img_intrinsic = np.load('/home/hoangqc/Datasets/AGZ/calibration_data/intrinsic_matrix.npy')
def_intrinsics = np.zeros((3, 4))
def_intrinsics[:3, :3] = img_intrinsic

mav_gt = np.genfromtxt(mav_gt_file, delimiter=',')[:, :10]
# Transform to MAV coordinate offset for visualization:
mav_gt = mav_gt - np.asarray([ 0,           mav_gt[0, 1], mav_gt[0, 2], mav_gt[0, 3],   # XYZ origin
                                0, 0, 0,    mav_gt[0, 7], mav_gt[0, 8], mav_gt[0, 9]])  # GPS origin

xyz,ypr = mav_gt[:, 1:4],mav_gt[:, 4:7]
transforms = get_transforms_mat44(xyz, ypr)
quads_colmap = get_transforms_quads_colmap(xyz, ypr)

# Get image data:
imgIDs = mav_gt[:,0].astype(int)
fileList = []
for id in range(mav_gt.shape[0]):
    fileName = '%05d' % imgIDs[id] + '.jpg'
    fileList.append(fileName)
    # print(id, str)



# imgFiles = [imgs[i] for i in imgIDs]

# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)
file = open('images.txt','w')
header = "# Image list with two lines of data per image:\n\
# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME"
file.write(header + '\n')
for i in range(mav_gt.shape[0]):
    data = []
    data.append(str(i+1))
    data.extend(quads_colmap[i].tolist())
    data.append(1)
    data.append(fileList[i])
    dumb = ' '.join(map(str,data))
    # print(dumb)
    file.write(dumb + '\n\n')
file.close()


# Update location
# pcd_xyz = o3d.geometry.PointCloud()
# pcd_xyz.points = o3d.utility.Vector3dVector(xyz)
# pcd_xyz.colors = o3d.utility.Vector3dVector(np.full(xyz.shape, [0, 255, 0]))
# pcd_gps = o3d.geometry.PointCloud()
# pcd_gps.points = o3d.utility.Vector3dVector(gps)
# pcd_gps.colors = o3d.utility.Vector3dVector(np.full(gps.shape, [255, 0, 0]))
#
# axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20.0, origin=[0, 0, 0])
# vis_list = []
# vis_list.append(pcd_xyz)
# vis_list.append(pcd_gps)
# vis_list.append(axis_pcd)
#
# for i in range(40):
#     cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=[0, 0, 0])
#     cam.transform(transforms[10+i*10])
#     vis_list.append(cam)
#     ...
# o3d.visualization.draw_geometries(vis_list)