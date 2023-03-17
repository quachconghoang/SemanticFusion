import os.path
from os import path
import glob
import numpy as np
import open3d as o3d
import cv2 as cv
import sys, copy
sys.path.append('../')
from scipy.spatial.transform import Rotation as R

from SlamUtils.transformation import pos_quats2SEs, pos_quats2SE_matrices, pose2motion, SEs2ses, line2mat
from SlamUtils.visualization import getVisualizationBB, getKeyframe
from SlamUtils.Loader.TartanAir import getRootDir, getDataSequences, getDataLists

import gtsam
from gtsam import Cal3_S2, Point3, Pose3

if __name__== "__main__":
    rootDIR = getRootDir()
    path = getDataSequences(root=rootDIR,scenario='office', level='Easy', seq_num=4)
    files_rgb_left, files_rgb_right, files_depth_left, poselist = getDataLists(dir=path, skip=10)

    assert (poselist.shape[1] == 7)  # x-y-z qx-qy-qz-qw
    poses_mat34 = pos_quats2SEs(poselist) # [R|t]
    poses_mat44 = pos_quats2SE_matrices(poselist)
    motions_mat = pose2motion(poses_mat34) # [R|t]
    motions_quat = SEs2ses(motions_mat).astype(np.float32) # x-y-z qx-qy-qz-qw

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Trajectory', width=1024, height=768)
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0,0,0])
    vis.add_geometry(axis_pcd)
    pose_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25, origin=[0,0,0])

    line_set = getVisualizationBB()
    vis.add_geometry(line_set)

    pointSet = o3d.geometry.PointCloud()
    vis.add_geometry(pointSet)

    num_keyframes = len(files_rgb_left)

    for id in range(num_keyframes):
        pose = poselist[id]
        pointSet.points.append([pose[0], pose[1], pose[2]])
        camFrame = getKeyframe(transform=poses_mat44[id])
        vis.add_geometry(camFrame)

        vis.update_geometry(pointSet)
        vis.update_geometry(camFrame)
        vis.poll_events()
        vis.update_renderer()

        # if id < num_keyframes-1:
        #     print(poses_mat34[id])

        #Read OpenCV ...||...||...||...||...
        rgb = cv.imread(files_rgb_left[id])
        cv.imshow('rgb', rgb)
        k = cv.waitKey(50)
        if k == 27:
            cv.destroyAllWindows()
            break

    cv.destroyAllWindows()
    vis.update_geometry(pointSet); vis.poll_events(); vis.update_renderer(); vis.run()
    vis.destroy_window()
