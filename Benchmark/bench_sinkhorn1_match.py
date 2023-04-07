import os
import sys
sys.path.append('../')

import numpy as np
import cv2 as cv
import ot
from scipy.optimize import linear_sum_assignment
from skimage import io
import json

from Semantics.SuperGlue.models.utils import (make_matching_plot_fast, frame2tensor)
from Semantics.image_proc_2D import matching, dnn_device, getSuperPoints_v2, getAnchorPoints
from Semantics.image_proc_3D import cal_Point_Project_General, generate3D, getGroundTruth,\
                        getPointsProject2D, getDisparityTartanAIR, get3DLocalFromDisparity

from config import gtsam, Point2, Point3, Rot3, Pose3, Cal3_S2, Values, PinholeCameraCal3_S2
from config import camera, RGBDImage, PointCloud,  Image

from SlamUtils.transformation import pos_quats2SEs, pos_quats2SE_matrices, pose2motion, SEs2ses, line2mat, tartan2kitti
from SlamUtils.Loader.TartanAir import rootDIR, getDataSequences, reload_with_MSCKF_Estimation,\
                                    getDataLists, tartan_camExtr

from Semantics.utils import camExtr, camIntr, getPoint3D

import pandas as pd

K = Cal3_S2(320, 320, 0.0, 320, 240)
inv_camExtr = np.linalg.inv(camExtr) # Cam to World

files_rgb_left, files_rgb_right, files_depth_left, poses_quad, poses_mat44 = [], [], [], [], []
pose_gt_mat44, pose_est_mat44 = [],[]

with open(rootDIR + 'tartanair_data.json', 'r') as fp:
    db = json.load(fp)

def getFrameInfo(id):
    frame = {   'color': io.imread(files_rgb_left[id]),
                'color_right': io.imread(files_rgb_right[id]),
                'depth': np.load(files_depth_left[id]),
                'transform': pose_gt_mat44[id],
                'pose_est': pose_est_mat44[id],
                'intr': camIntr,
                'extr': camExtr     }
    return frame

def evalScores(pts0,pts1,matches, kp0_gt,kp0_valid, thresh=.9):
    # kpt0 = pts0['pts'].astype(int)
    kpt1 = pts1['pts'].astype(int)
    match_state = []
    for id0, id1, mVal in matches:
        # p0 = kpt0[int(id0)]
        p0_gt = kp0_gt[int(id0)]
        valid = kp0_valid[int(id0)]
        p1 = kpt1[int(id1)]

        dis = np.linalg.norm(p0_gt - p1)
        if (mVal< thresh):
            if(dis < 8):
                match_state.append('TRUE')
            else:
                match_state.append('FALSE')
        else:
            if valid:
                match_state.append('SKIP_BUT_VALID')
            else:
                match_state.append('SKIP')

    precision, recall, f1 = 0,0,0
    if(match_state.count('TRUE')>0):
        precision = match_state.count('TRUE')/(match_state.count('TRUE') + match_state.count('FALSE') )
        recall = match_state.count('TRUE')/( match_state.count('TRUE') + match_state.count('SKIP_BUT_VALID') )
        f1 = 2*precision*recall/(precision+recall)
    else:
        return 0,0,0
    # print('Precision = ', precision, '; Recall = ', recall, '; F1 = ', f1)
    return precision, recall, f1

def getMatchPrecision(src_id, tar_id):
    source = getFrameInfo(src_id)
    target = getFrameInfo(tar_id)
    motion = np.linalg.inv(source['pose_est']).dot(target['pose_est'])

    generate3D(source)
    target_cam = PinholeCameraCal3_S2(Pose3(target['transform']), K)

    pts0 = getSuperPoints_v2(cv.cvtColor(source['color'], cv.COLOR_RGB2GRAY))
    pts1 = getSuperPoints_v2(cv.cvtColor(target['color'], cv.COLOR_RGB2GRAY))
    desc0 = pts0['desc']
    desc1 = pts1['desc']

    norm_self_src = np.sqrt(2 - 2 * np.clip(np.dot(desc0.T, desc0), -1, 1))
    norm_self_tar = np.sqrt(2 - 2 * np.clip(np.dot(desc1.T, desc1), -1, 1))
    norm_cross = np.sqrt(2 - 2 * np.clip(np.dot(desc0.T, desc1), -1, 1))

    # get stereo map
    src_disparity = getDisparityTartanAIR(source['color'], source['color_right'])

    kp0 = pts0['pts']
    kp1 = pts1['pts']
    kp0_3d = get3DLocalFromDisparity(kp0, src_disparity)

    num_detected = pts0['pts'].shape[0]
    kp0_gt, kp0_valid = getPointsProject2D(pts_2d=pts0['pts'], pts_3d=source['point3D'], target_cam=target_cam)

    if(  (len(kp0_gt)<5) | (kp0_valid.count(True)<10) | (len(kp0)<10) |(len(kp1)<10) ) :
        # print('Impossible match')
        return 0, 0, 0

    cam_target_predicted = PinholeCameraCal3_S2(Pose3(motion), K)

    extr = camExtr[:3, :3]  # for convert global point to cam points  -> z-forward
    inv_extr = inv_camExtr[:3, :3]  # for convert cam (z-forward) to NED -> z-down

    kp0_expect = []
    for pt3d in kp0_3d:
        pt2d = cal_Point_Project_General(cam_target_predicted, pt3d, extr)
        kp0_expect.append(pt2d)
    kp0_expect = np.asarray(kp0_expect)

    # gen distance-table
    map = np.zeros(shape=[kp0.shape[0], kp1.shape[0]])
    std = 64
    for i, p0_e in enumerate(kp0_expect):
        for j, p1 in enumerate(kp1):
            map[i, j] = np.sqrt(np.linalg.norm(p0_e - p1) / std)

    ### Get Sinkhorn Map
    a, b = [], []
    for pt_his in norm_self_src:
        a.append(pt_his.mean() - 1)
    for pt_his in norm_self_tar:
        b.append(pt_his.mean() - 1)


    Gs = ot.sinkhorn(a, b, norm_cross, reg=1e-1, numItermax=20)
    ### Sinkhorn with 3D project map
    match_id = linear_sum_assignment(cost_matrix = (1-Gs)+map)
    match_score = norm_cross[match_id]
    matches = np.stack((match_id[0], match_id[1], match_score), axis=1)

    # Motion check ...
    motion = Pose3(motion)
    rpy = motion.rotation().rpy()
    xyz = motion.translation()
    r_cost = np.linalg.norm(rpy)
    t_cost = np.linalg.norm(xyz)
    # print('Transform:', r_cost, t_cost)
    if ((r_cost < 0.25) & (t_cost < 1.2)):
        return evalScores(pts0, pts1, matches, kp0_gt, kp0_valid, thresh=1.2)

    return evalScores(pts0, pts1, matches, kp0_gt, kp0_valid, thresh=1.1)

def traceSquence(sce,lvl,traj):
    print_name = sce + '_' + lvl + '_' + traj
    step = 10
    src_id = 0
    last_id = len(files_rgb_left)
    total_rs = []

    while (src_id + 20) < last_id:
        # print(src_id)
        p4, r4, f4 = getMatchPrecision(src_id, src_id + 20)
        rs = [p4, r4, f4]
        total_rs.append(rs)
        src_id += step

    rs_np = np.asarray(total_rs)
    p4 = rs_np[:, 0]
    r4 = rs_np[:, 1]
    f4 = rs_np[:, 2]

    print(print_name, '%6f' % p4[p4 != 0].mean(), '%6f' % r4[r4 != 0].mean(), '%6f' % f4[f4 != 0].mean())


keys = db['keys']
levels = db['levels']
sce = 'abandonedfactory'
for lv in levels:
    trajs = db[sce][lv]
    for traj in trajs:
        path = os.path.join(rootDIR, sce, lv, traj, '')
        save_dir = os.path.join(rootDIR, '..', 'TartanAir_Bag', sce, lv, traj, '')
        files_rgb_left, files_rgb_right, files_depth_left, poses_quad = getDataLists(dir=path)
        poses_mat44 = pos_quats2SE_matrices(poses_quad)
        if os.path.exists(os.path.join(save_dir, 'pose_est.txt')) == False:
            continue  # SKIP ...

        pose_est_mat44, pose_gt_mat44, \
        files_rgb_left, files_rgb_right, \
        files_depth_left = reload_with_MSCKF_Estimation(files_rgb_left, files_rgb_right, \
                                                        files_depth_left, save_dir=save_dir)
        traceSquence(sce, lv, traj)
    print('----------')


# sce = 'abandonedfactory'
# lv = 'Hard'
# trajs = db[sce][lv]
# for traj in trajs:
#     path = os.path.join(rootDIR, sce, lv, traj, '')
#     save_dir = os.path.join(rootDIR, '..', 'TartanAir_Bag', sce, lv, traj, '')
#     files_rgb_left, files_rgb_right, files_depth_left, poses_quad = getDataLists(dir=path)
#     poses_mat44 = pos_quats2SE_matrices(poses_quad)
#     if os.path.exists(os.path.join(save_dir, 'pose_est.txt')) == False:
#         continue  # SKIP ...
#
#     pose_est_mat44, pose_gt_mat44, \
#     files_rgb_left, files_rgb_right, \
#     files_depth_left = reload_with_MSCKF_Estimation(files_rgb_left, files_rgb_right, \
#                                                     files_depth_left, save_dir=save_dir)
#     traceSquence(sce, lv, traj)