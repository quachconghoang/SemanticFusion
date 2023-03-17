import os.path
from os import path
import glob
import numpy as np
import open3d as o3d
from functools import partial
import cv2 as cv
import sys
sys.path.append('../')

from SlamUtils.transformation import pos_quats2SEs, pos_quats2SE_matrices, pose2motion, SEs2ses, line2mat, tartan2kitti
from SlamUtils.visualization import getVisualizationBB, getKeyframe
from SlamUtils.utils import dataset_intrinsics
from SlamUtils.Loader.TartanAir import getRootDir, getDataSequences, getDataLists

from GTSAM.keyframe import Keyframe

if sys.platform =='linux':
    import gtsam
    from gtsam import Cal3_S2, Point3, Pose3

from Semantics.dnn_engine import DnnEngine

engine = DnnEngine()

def saveView(viz):
    viz.capture_screen_image(path)
    ctr = viz.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters("NED-View.json", param)
    print("Saved")


rootDIR = getRootDir()
path = getDataSequences(root=rootDIR, scenario='office', level='Easy', seq_num=4)
files_rgb_left, files_rgb_right, files_depth_left, poselist = getDataLists(dir=path, skip=10)

############ GTSAM ###################
focalx, focaly, centerx, centery = dataset_intrinsics(dataset='tartanair')
# K = Cal3_S2(focalx, focaly, 0.0, centerx, centery)
# Add a prior on pose x1. This indirectly specifies where the origin is.
# 0.3 rad std on roll,pitch,yaw and 0.1m on x,y,z
# pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))


assert (poselist.shape[1] == 7)  # x-y-z qx-qy-qz-qw
poses_mat34 = pos_quats2SEs(poselist)  # [R|t - array 12]
poses_mat34_kitty, poses_mat44_kitty = tartan2kitti(poselist)
poses_mat44 = pos_quats2SE_matrices(poselist)
motions_mat = pose2motion(poses_mat34)  # [R|t]
motions_quat = SEs2ses(motions_mat).astype(np.float32)  # x-y-z qx-qy-qz-qw

line_set = getVisualizationBB()
pointSet = o3d.geometry.PointCloud()
blob3D = o3d.geometry.PointCloud()

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name='Trajectory', width=1024, height=768)
vis.register_key_callback(ord("S"), partial(saveView))

axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
vis.add_geometry(axis_pcd)


pose_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25, origin=[0, 0, 0])
vis.add_geometry(line_set)
vis.add_geometry(pointSet)
vis.add_geometry(blob3D)

# param = o3d.io.read_pinhole_camera_parameters(rootDIR + 'NED-View.json')
# vis.get_view_control().convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
vis.poll_events();    vis.update_renderer()

num_keyframes = len(files_rgb_left)
keyframes = []
num_keyframes = 60

for id in range(num_keyframes):
    pose = poselist[id]
    pointSet.points.append([pose[0], pose[1], pose[2]])
    k = Keyframe(rgb=cv.imread(files_rgb_left[id]), depth=np.load(files_depth_left[id]), transform=poses_mat44[id],_processing=True)
    k.kp2D = engine.getSuperPoint(k.gray_raw)
    k.loadKeyCloud()
    blob3D += k.keyCloud
    vis.update_geometry(blob3D)

    camFrame = getKeyframe(transform=poses_mat44[id],color=[1,0,0])

    vis.add_geometry(camFrame)
    vis.update_geometry(pointSet)
    vis.update_geometry(camFrame)
    vis.poll_events()
    vis.update_renderer()

    if id < num_keyframes - 1:
        print(motions_quat[id][0:3])

    # rgb = cv.imread(files_rgb_left[id])
    # depth = np.load(files_depth_left[id])
    cv.imshow('rgb', k.color_raw)
    cv.imshow('depth', k.depth_raw/10)
    k = cv.waitKey(0)
    if k == 27:
        cv.destroyAllWindows()
        break

cv.destroyAllWindows()
vis.poll_events();vis.update_renderer();vis.run()
# ctr = vis.get_view_control()
# param = ctr.convert_to_pinhole_camera_parameters()
# o3d.io.write_pinhole_camera_parameters("NED-View.json", param)
vis.destroy_window()
# visFPV.poll_events();visFPV.update_renderer();visFPV.run()



### STUPID - STEPS - to Fuck C++:
# I.    Initialize local group (Local BA)
#       0 - Define Keyframe specification:
#           + Point - Lines - Semantics Descriptor
#           + Raw rgb - depth - calibration
#           + GLOBAL - pose
#       1 - Extract features & matching (Adopt deeplearning matching)
#       2 - Local SFM with points - lines constraints: init point-cloud & 3D-to-2D projection with GTSAM (Factor Graph with GTSAM)
#           I'm considering other non-linear solver such as: G2O, Ceres and SE-Sync; but it is not a good choice at all.
#           (Pure Python research tools are trending)
# II.   Incremental local BA (Forget) -> Global BA: go straight with iSAM2
#       1 - KDTree (Octree) keyframes (to generate smart - landmarks)
#           For example: https://stackoverflow.com/questions/65003877/understanding-leafsize-in-scipy-spatial-kdtree
#       2 - Insert procedure: matching score / entropy score / minimum distance / timestamp
#         - Terminate procedure?
#       3 - Develop some interesting Smart Factor from Semantic Graph: Points & Line segments
#           Examples: 3D Bounding-Box, Dual-Quadratic.
#       4 - iSAM2 Smart-Factors (SEMANTIC_Factor)
