import os.path
from os import path
import glob
import numpy as np
import open3d as o3d
from functools import partial
import cv2 as cv
from skimage import data, io, filters
import sys
sys.path.append('../')

from SlamUtils.transformation import pos_quats2SEs, pos_quats2SE_matrices, pose2motion, SEs2ses, line2mat, tartan2kitti, quads_NED_to_ENU
from SlamUtils.visualization import getVisualizationBB, getKeyframe
from SlamUtils.utils import dataset_intrinsics
from SlamUtils.Loader.TartanAir import getRootDir, getDataSequences, getDataLists, tartan_camExtr, ros_camExtr

from open3d import camera
from open3d.cuda.pybind.geometry import RGBDImage,PointCloud
from open3d.cuda.pybind.utility import Vector3dVector

from Semantics.image_proc_2D import getSobelMask,getDistanceMask

thresh_sobel = 30 #Soft thresh
thresh_distace = 8
# voxel_size = 0.025 # indoor
voxel_size = 0.05 # outdoor

rootDIR = getRootDir()
path = getDataSequences(root=rootDIR, scenario='office', level='Easy', seq_num=4)
# path = getDataSequences(root=rootDIR, scenario='neighborhood', level='Easy', seq_num=0)
# path = getDataSequences(root=rootDIR, scenario='seasidetown', level='Easy', seq_num=0)
files_rgb_left, files_rgb_right, files_depth_left, poselist = getDataLists(dir=path, skip=5)

# NED to ROS ENU:
# body-fixed NED → ROS ENU: (x y z)→(x -y -z) or (w x y z)→(x -y -z w)
# local NED → ROS ENU: (x y z)→(y x -z) or (w x y z)→(y x -z w)


poselist_ENU = np.array([quads_NED_to_ENU(q) for q in poselist])


# path = getDataSequences(root=rootDIR, scenario='office', level='Easy', seq_num=0)
# _rgb_left, _rgb_right, _depth_left, _poselist = getDataLists(dir=path, skip=5)
# files_rgb_left.extend(_rgb_left)
# files_rgb_right.extend(_rgb_right)
# files_depth_left.extend(_depth_left)
# poselist= np.concatenate((poselist, _poselist), axis=0)

focalx, focaly, centerx, centery = dataset_intrinsics(dataset='tartanair')

poses_mat34 = pos_quats2SEs(poselist)  # [R|t - array 12]
# poses_mat34_kitty, poses_mat44_kitty = tartan2kitti(poselist)
poses_mat44 = pos_quats2SE_matrices(poselist)
poses_mat44_ENU = pos_quats2SE_matrices(poselist_ENU)
motions_mat = pose2motion(poses_mat34)  # [R|t]
motions_quat = SEs2ses(motions_mat).astype(np.float32)  # x-y-z qx-qy-qz-qw

config_cam = {'width':640, 'height':480, 'fx':320, 'fy':320, 'cx':320, 'cy':240}
cameraIntrinsic = camera.PinholeCameraIntrinsic(**config_cam) #open3d if need
# cameraIntrinsic = camera.PinholeCameraIntrinsic(width=640, height=480, fx=320, fy=320, cx=320, cy=240)

# 2D TESTING
# id = 0
# img_color = io.imread(files_rgb_left[id])
# img_depth = np.load(files_depth_left[id])
# img_trans = poses_mat44[id]
# img_rgbd = RGBDImage.create_from_color_and_depth(
#                 color=o3d.geometry.Image(img_color),
#                 depth=o3d.geometry.Image(img_depth),
#                 depth_scale=1.0, depth_trunc=np.inf,
#                 convert_rgb_to_intensity=False)
# ----- get cloud -----
# img_cloud = PointCloud.create_from_rgbd_image(image=img_rgbd,
#                                                 intrinsic=cameraIntrinsic,
#                                                 extrinsic=T)
# img_cloud.transform(img_trans)
# ds_cloud = img_cloud.voxel_down_sample(voxel_size=0.01)
# o3d.visualization.draw_geometries([img_cloud])

# ----- GET SOBEL MASKS =)) -----
# img_gray = cv.cvtColor(img_color,cv.COLOR_RGB2GRAY)
# img_pcd_mask = getSobelMask(img_gray,thesh_sobel)
# io.imshow(img_mask); io.show()


# ====================================================
#3D VISUALIZATION
vizBB = getVisualizationBB()
pointSet = o3d.geometry.PointCloud()
blob3D = o3d.geometry.PointCloud()
global_pcd = o3d.geometry.PointCloud()

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name='Trajectory', width=1024, height=768)

axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
vis.add_geometry(vizBB)
vis.add_geometry(axis_pcd)
vis.add_geometry(blob3D)

vis.poll_events();vis.update_renderer()


for id in range(len(files_rgb_left)):
    print(id, blob3D.points.__len__())
    # pose = poselist[id]
    img_color = io.imread(files_rgb_left[id])
    img_depth = np.load(files_depth_left[id])
    img_trans = poses_mat44_ENU[id]
    img_rgbd = RGBDImage.create_from_color_and_depth(
        color=o3d.geometry.Image(img_color),
        depth=o3d.geometry.Image(img_depth),
        depth_scale=1.0, depth_trunc=np.inf,
        convert_rgb_to_intensity=False)
    img_cloud = PointCloud.create_from_rgbd_image(image=img_rgbd,
                                                  intrinsic=cameraIntrinsic,
                                                  extrinsic=ros_camExtr)
    img_cloud.transform(img_trans)

    # Filtering: Sobel & Depth
    img_gray = cv.cvtColor(img_color, cv.COLOR_RGB2GRAY)

    mask_sobel = getSobelMask(img_gray,thresh=thresh_sobel)
    mask_depth = getDistanceMask(img_depth, thresh=thresh_distace)
    pcd_mask = mask_depth & mask_sobel

    tmp_cloud = o3d.geometry.PointCloud()
    tmp_cloud.points = Vector3dVector(np.asarray(img_cloud.points)[pcd_mask])
    tmp_cloud.colors = Vector3dVector(np.asarray(img_cloud.colors)[pcd_mask])

    # tmp_cloud.transform(img_trans)

    global_pcd += tmp_cloud
    global_pcd = global_pcd.voxel_down_sample(voxel_size=voxel_size)

    # tmp_cloud = tmp_cloud.voxel_down_sample(voxel_size=0.03)
    # blob3D.points += tmp_cloud.points
    # blob3D.colors += tmp_cloud.colors
    blob3D.points = global_pcd.points
    blob3D.colors = global_pcd.colors
    # blob3D = blob3D.voxel_down_sample(voxel_size=0.03)

    vis.update_geometry(blob3D)
    vis.poll_events()
    vis.update_renderer()

vis.poll_events();vis.update_renderer();vis.run()
vis.destroy_window()

# o3d.io.write_point_cloud("cloud.ply", global_pcd, write_ascii=True, compressed=False, print_progress=True)
