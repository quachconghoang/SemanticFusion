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

import time

K = Cal3_S2(320, 320, 0.0, 320, 240)
inv_camExtr = np.linalg.inv(camExtr) # Cam to World

files_rgb_left, files_rgb_right, files_depth_left, poses_quad, poses_mat44 = [], [], [], [], []
pose_est_mat44 = []

with open(rootDIR + 'tartanair_data.json', 'r') as fp:
    db = json.load(fp)

def getFrameInfo(id):
    frame = {   'color': io.imread(files_rgb_left[id]),
                'color_right': io.imread(files_rgb_right[id]),
                'depth': np.load(files_depth_left[id]),
                'transform': poses_mat44[id],
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
        precision = match_state.count('TRUE')/(match_state.count('TRUE')+match_state.count('FALSE'))
        recall = match_state.count('TRUE')/(match_state.count('TRUE')+match_state.count('SKIP_BUT_VALID'))
        f1 = 2*precision*recall/(precision+recall)
    # print('Precision = ', precision, '; Recall = ', recall, '; F1 = ', f1)
    return precision, recall, f1

def getMatchPrecision_Anchor(src_id, tar_id):
    # print(src_id, ' -> ', tar_id)
    source = getFrameInfo(src_id) #(30 * 5)  # 32*5
    target = getFrameInfo(tar_id) #(34 * 5)  # 36*5
    generate3D(source)
    src_p3d = source['point3D']

    source_cam = PinholeCameraCal3_S2(Pose3(source['transform']), K)
    target_cam = PinholeCameraCal3_S2(Pose3(target['transform']), K)

    src_gray = cv.cvtColor(source['color'], cv.COLOR_RGB2GRAY)
    tar_gray = cv.cvtColor(target['color'], cv.COLOR_RGB2GRAY)
    src_disparity = getDisparityTartanAIR(source['color'], source['color_right'])
    tar_disparity = getDisparityTartanAIR(target['color'], target['color_right'])

    pts0 = getSuperPoints_v2(src_gray)
    pts1 = getSuperPoints_v2(tar_gray)

    kp0 = pts0['pts']
    kp1 = pts1['pts']
    desc0 = pts0['desc']
    desc1 = pts1['desc']

    norm_self_src = np.sqrt(2 - 2 * np.clip(np.dot(desc0.T, desc0), -1, 1))
    norm_self_tar = np.sqrt(2 - 2 * np.clip(np.dot(desc1.T, desc1), -1, 1))
    his_src = np.mean(norm_self_src, axis=0) / 2
    his_tar = np.mean(norm_self_tar, axis=0) / 2

    norm_cross = np.sqrt(2 - 2 * np.clip(np.dot(desc0.T, desc1), -1, 1))
    kp0_3d = get3DLocalFromDisparity(kp0, src_disparity)
    kp1_3d = get3DLocalFromDisparity(kp1, tar_disparity)

    # Bench ...
    kpt0_gt, kpt0_valid = getGroundTruth(src_p3d_world=src_p3d,
                                         src_p2d=kp0,
                                         target_cam=target_cam)

    if(  (len(kpt0_gt)==0) | (kpt0_valid.count(True)==0) | (len(kp0)<10) |(len(kp1)<10) ) :
        # print('Impossible match')
        return 0, 0, 0

    st = time.time()
    # Cross check = BAD -> Sinkhorn CROSS CHECK
    matches, Gs = getAnchorPoints(norm_cross, his_src, his_tar, sorting=True)
    if(matches.shape[0] < 5):
        print('Anchoring number too small = ',matches.shape[0])
        return 0,0,0
    # evalScores(pts0, pts1, matches, kpt0_gt, kpt0_valid, thresh=.9)

    anchor_src_id = matches.T[0].astype(int)
    anchor_tar_id = matches.T[1].astype(int)

    src_anchor = kp0[anchor_src_id]
    src_anchor_3D = kp0_3d[anchor_src_id]
    tar_anchor = kp1[anchor_tar_id]
    tar_anchor_3D = kp1_3d[anchor_tar_id]

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    s, r, t, x = cv.solvePnPRansac(objectPoints=src_anchor_3D, imagePoints=tar_anchor,
                                   cameraMatrix=camIntr.intrinsic_matrix, distCoeffs=None)

    if(s == False):
        print('Anchoring failed !!!')
        return 0,0,0

    kp0_expected, jacobian = cv.projectPoints(kp0_3d, r, t, camIntr.intrinsic_matrix, dist_coeffs)

    et = time.time()
    stage_1 = et - st

    map = np.zeros(shape=[kp0.shape[0], kp1.shape[0]])
    std = 64
    for i, p0_e in enumerate(kp0_expected):
        for j, p1 in enumerate(kp1):
            map[i, j] = np.sqrt(np.linalg.norm(p0_e - p1) / std)

    st = time.time()

    match_id = linear_sum_assignment(cost_matrix=(1 - Gs) + map)
    match_score = norm_cross[match_id]
    matches = np.stack((match_id[0], match_id[1], match_score), axis=1)

    et = time.time()
    stage_2 = et - st
    # print(kp0.shape[0] + kp1.shape[0])
    print('time = \t', stage_2 + stage_1, stage_1, stage_2)
    return evalScores(pts0, pts1, matches, kpt0_gt, kpt0_valid, thresh=1.2)

def traceSquence(sce,lvl,traj):
    # path = os.path.join(rootDIR, sce, lvl, traj, '')
    # print(path)
    print_name = sce + '_' + lvl + '_' + traj

    step = 2
    src_id = 0
    last_id = len(files_rgb_left)
    total_rs = []

    while (src_id + 4) < last_id:
        # print(src_id, '->', src_id+4)
        p4, r4, f4 = getMatchPrecision_Anchor(src_id, src_id + 4)
        rs = [p4, r4, f4]
        total_rs.append(rs)
        src_id += step

    rs_np = np.asarray(total_rs)
    p4 = rs_np[:, 0]; r4 = rs_np[:, 1];  f4 = rs_np[:, 2]
    print(print_name, '%6f' % p4[p4 != 0].mean(), '%6f' % r4[r4 != 0].mean(), '%6f' % f4[f4 != 0].mean())
    ...


keys = db['keys']
levels = db['levels']
est_rs = {}
names = []
f1 = []
sce = 'hospital'

# for lv in levels:
#     trajs = db[sce][lv]
#     for traj in trajs:
#         path = os.path.join(rootDIR, sce, lv, traj, '')
#         files_rgb_left, files_rgb_right, files_depth_left, poses_quad = getDataLists(dir=path)
#         poses_mat44 = pos_quats2SE_matrices(poses_quad)
#         traceSquence(sce, lv, traj)
#     print('----------')

# lv = 'Hard'
# trajs = db[sce][lv]
# for traj in trajs:
#     path = os.path.join(rootDIR, sce, lv, traj, '')
#     # save_dir = os.path.join(rootDIR, '..', 'TartanAir_Bag', sce, lv, traj, '')
#     files_rgb_left, files_rgb_right, files_depth_left, poses_quad = getDataLists(dir=path, skip=5)
#     poses_mat44 = pos_quats2SE_matrices(poses_quad)
#     traceSquence(sce, lv, traj)
# print('----------')


sce = 'carwelding'
lv = 'Hard'
traj = 'P002'
path = os.path.join(rootDIR, sce, lv, traj, '')
files_rgb_left, files_rgb_right, files_depth_left, poses_quad = getDataLists(dir=path, skip=5)
poses_mat44 = pos_quats2SE_matrices(poses_quad)
traceSquence(sce, lv, traj)