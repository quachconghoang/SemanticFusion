import os.path
from os import path
import glob
import numpy as np
import open3d as o3d
import cv2 as cv
import sys

from pathlib import Path
import datetime

from SlamUtils.transformation import pos_quats2SEs, pos_quats2SE_matrices, pose2motion, SEs2ses, line2mat, tartan2kitti
from config import TartanAir_rootDIRS, TartanAir_scenarios, TartanAir_levels

tartan_camExtr = np.array([[0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [1, 0, 0, 0],
                            [0, 0, 0, 1]], dtype=np.float32)


def getRootDir():
    for dir in TartanAir_rootDIRS:
        if path.exists(dir):
            print("directory exists: ", dir)
            return dir

rootDIR = getRootDir()

def getDataSequences(root=rootDIR, scenario='neighborhood', level='Easy', seq_num=0):
    TartanAir_scenarios = os.listdir(root)
    path_scr = root
    if any(scenario in s for s in TartanAir_scenarios):
        path_scr += (scenario + '/' + level + '/')
    else:
        print('loading error at scenario: ', scenario);return None

    _trajs = os.listdir(path_scr)
    _trajs = list(filter(lambda x: 'P' in x, _trajs))
    _trajs.sort()

    if seq_num < len(_trajs):
        path_scr += (_trajs[seq_num]+'/')
        print('loading path: ',path_scr)
        return path_scr
    else:
        print('loading error at seq: ', seq_num)
        return None

def getDataLists(dir='', skip=1):
    # Load Left-right-GroundTruth
    left_dir = dir + 'image_left/'
    right_dir = dir + 'image_right/'
    files_rgb_left, files_depth_left, files_rgb_right = [], [], []

    if (os.path.exists(left_dir)):
        files_rgb_left = glob.glob(dir + 'image_left/' + '*.png'); files_rgb_left.sort()
        files_depth_left = glob.glob(dir + 'depth_left/' + '*.npy'); files_depth_left.sort()

    if(os.path.exists(right_dir)):
        files_rgb_right = glob.glob(dir + 'image_right/' + '*.png');files_rgb_right.sort()
    poselist = np.loadtxt(dir + 'pose_left.txt').astype(np.float32)

    if skip > 1:
        return files_rgb_left[0::skip], files_rgb_right[0::skip], files_depth_left[0::skip], poselist[0::skip]
    return files_rgb_left, files_rgb_right, files_depth_left, poselist

### LOAD ESTIMATION ###
def reload_with_MSCKF_Estimation(rgb_left, rgb_right, depth_left,
                                 msckf_dir = str(Path.home())+'/catkin_ws/tmp/sim/',
                                 seq_name = 'office_004'):
    # seq_name = 'office_004'
    # save_dir = str(Path.home()) + '/catkin_ws/tmp/sim/' + 'office_004' + '/'
    save_dir = msckf_dir + seq_name + '/'
    # Load Raw
    pose_est = np.loadtxt(fname = save_dir + 'pose_est.txt')[:,:8]
    pose_gt = np.loadtxt(fname = save_dir + 'pose_gt.txt')[:,:8]
    data_ids = np.loadtxt(fname = save_dir + 'data_id.txt').astype(np.int32)

    # Trace start id in estimation
    base_time = pose_gt[0][0]
    est_time = pose_est[0][0]
    start_id = np.round((est_time - base_time)*1000/50).astype(np.int32)

    # remove time_stamp
    pose_est = pose_est[:,1:]
    pose_gt = pose_gt[:,1:]
    pose_gt_mat44 = pos_quats2SE_matrices(pose_gt)

    # remap to global pose
    pose_ref = pose_gt_mat44[start_id-1]
    pose_est_mat44 = pos_quats2SE_matrices(pose_est)
    for i,pose in enumerate(pose_est_mat44):
        pose_est_mat44[i] = pose_ref.dot(pose)

    # remap to estimation to gt
    est_ids = [*range(start_id, start_id + pose_est.shape[0])]
    pose_gt = pose_gt[est_ids]
    pose_gt_mat44 = pos_quats2SE_matrices(pose_gt)

    # remap to tartanAIR
    data_ids = data_ids[est_ids]
    rgb_left = [rgb_left[i] for i in data_ids]
    depth_left = [depth_left[i] for i in data_ids]
    rgb_right = [rgb_right[i] for i in data_ids]

    return pose_est_mat44, pose_gt_mat44, rgb_left,rgb_right,depth_left


def getVisualizationBB(maxX=10, maxY=10, maxZ=2, minX=-10, minY=-10, minZ=-2):
    box_points = [[minX, minY, minZ], [maxX, minY, minZ], [minX, maxY, minZ], [maxX, maxY, minZ],
              [minX, minY, maxZ], [maxX, minY, maxZ], [minX, maxY, maxZ], [maxX, maxY, maxZ]]
    box_lines = [[0, 1],[0, 2],[1, 3],[2, 3],[4, 5],[4, 6],[5, 7],[6, 7],[0, 4],[1, 5],[2, 6],[3, 7],]
    box_colors = [[1, 0, 0] for i in range(len(box_lines))]
    line_set = o3d.geometry.LineSet( points=o3d.utility.Vector3dVector(box_points), lines=o3d.utility.Vector2iVector(box_lines))
    line_set.colors = o3d.utility.Vector3dVector(box_colors)
    return line_set
