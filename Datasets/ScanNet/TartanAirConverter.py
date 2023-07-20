import os
import sys
sys.path.append('../')

import numpy as np
import cv2 as cv
from skimage import io
import json
from matplotlib import pyplot as plt

from config import gtsam, Point2, Point3, Rot3, Pose3, Cal3_S2, Values, PinholeCameraCal3_S2
from config import camera, RGBDImage, PointCloud,  Image

from SlamUtils.transformation import pos_quats2SEs, pos_quats2SE_matrices, pose2motion, SEs2ses, line2mat, tartan2kitti, quads_NED_to_ENU
from SlamUtils.Loader.TartanAir import rootDIR, getDataSequences, getDataLists
from SlamUtils.visualization import getVisualizationBB, getKeyframe


import open3d as o3d

K = Cal3_S2(320, 320, 0.0, 320, 240)
camExtr = np.array([[0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1]], dtype=np.float64)
camIntr = camera.PinholeCameraIntrinsic(width=640, height=480, fx=320, fy=320, cx=320, cy=240)


files_rgb_left, files_rgb_right, files_depth_left, poses_quad, poses_mat44 = [], [], [], [], []

with open(rootDIR + 'tartanair_data.json', 'r') as fp:
    db = json.load(fp)


path = os.path.join(rootDIR,'office','Easy', 'P004', '')
files_rgb_left, files_rgb_right, files_depth_left, poselist = getDataLists(dir=path,skip=5)
poselist_ENU = np.array([quads_NED_to_ENU(q) for q in poselist])

poses_mat44 = pos_quats2SE_matrices(poselist)
poses_mat44_ENU = pos_quats2SE_matrices(poselist_ENU)

axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

id = 0
img_trans = poses_mat44_ENU[id]

img_color = io.imread(files_rgb_left[id])
img_depth = np.load(files_depth_left[id])

img_rgbd = RGBDImage.create_from_color_and_depth(
    color=o3d.geometry.Image(img_color),
    depth=o3d.geometry.Image(img_depth),
    depth_scale=1.0, depth_trunc=40,
    convert_rgb_to_intensity=False)

img_cloud = PointCloud.create_from_rgbd_image(image=img_rgbd, intrinsic=camIntr, extrinsic=camExtr)
img_cloud.transform(img_trans)
camera_vis = [axis_pcd,img_cloud]

for pose in poses_mat44_ENU:
    camFrame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    camFrame.transform(pose)
    camera_vis.append(camFrame)


o3d.visualization.draw_geometries(camera_vis[0:10])