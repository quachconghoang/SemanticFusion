import os
import sys
sys.path.append('../')

import numpy as np
import cv2 as cv
from skimage import io
import json
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
plt.rcParams['figure.dpi'] = 300

from Semantics.SuperGlue.models.utils import frame2tensor
from Semantics.image_proc_2D import matching, dnn_device
from Semantics.image_proc_3D import cal_Point_Project_TartanCam

from config import gtsam, Point2, Point3, Rot3, Pose3, Cal3_S2, Values, PinholeCameraCal3_S2
from config import camera, RGBDImage, PointCloud,  Image

from SlamUtils.transformation import pos_quats2SEs, pos_quats2SE_matrices, pose2motion, SEs2ses, line2mat, tartan2kitti
from SlamUtils.Loader.TartanAir import rootDIR, getDataSequences, getDataLists, tartan_camExtr

from Semantics.utils import camExtr, camIntr, getPoint3D


K = Cal3_S2(320, 320, 0.0, 320, 240)
files_rgb_left, files_rgb_right, files_depth_left, poses_quad, poses_mat44 = [], [], [], [], []

with open(rootDIR + 'tartanair_data.json', 'r') as fp:
    db = json.load(fp)

path = os.path.join(rootDIR,'office','Easy', 'P004', '')

files_rgb_left, files_rgb_right, files_depth_left, poselist = getDataLists(dir=path, skip=5)
poses_mat44 = pos_quats2SE_matrices(poselist)

def getFrameInfo(id):
    frame = {   'color': io.imread(files_rgb_left[id]),
                'color_right': io.imread(files_rgb_right[id]),
                'depth': np.load(files_depth_left[id]),
                'transform': poses_mat44[id],
                'intr': camIntr,
                'extr': camExtr     }
    return frame

def getMatchPrecision_Hungarian(source_id,target_id):
    source = getFrameInfo(source_id)
    target = getFrameInfo(target_id)
    source_cam = PinholeCameraCal3_S2(Pose3(source['transform']), K)
    target_cam = PinholeCameraCal3_S2(Pose3(target['transform']), K)

    src_gray = cv.cvtColor(source['color'], cv.COLOR_RGB2GRAY)
    src_p3d = getPoint3D(source)
    tar_gray = cv.cvtColor(target['color'], cv.COLOR_RGB2GRAY)

    # FRAME 0
    frame_tensor = frame2tensor(src_gray, dnn_device)
    keys = ['keypoints', 'scores', 'descriptors']
    last_data = matching.superpoint({'image': frame_tensor})
    last_data = {k + '0': last_data[k] for k in keys}
    last_data['image0'] = frame_tensor
    last_frame = src_gray

    # FRAME 1
    frame_tensor_tar = frame2tensor(tar_gray, dnn_device)
    current_data = matching.superpoint({'image': frame_tensor_tar})
    current_data = {k + '1': current_data[k] for k in keys}
    current_data['image1'] = frame_tensor_tar

    kpts0 = last_data['keypoints0'][0].cpu().numpy()
    kpts1 = current_data['keypoints1'][0].cpu().numpy()
    desc0 = last_data['descriptors0'][0].cpu().numpy()
    desc1 = current_data['descriptors1'][0].cpu().numpy()

    kpt0_gt = []
    kpt0_valid = []

    for kp in kpts0:
        x, y = kp
        p0w = Point3(src_p3d[int(y * 640 + x)])
        p0_target, d_prj = cal_Point_Project_TartanCam(target_cam, p0w)
        kpt0_gt.append(p0_target)
        if (0 < p0_target[0] < 640) & (0 < p0_target[1] < 480):
            d_gt = target['depth'][p0_target[1], p0_target[0]]
            dif = abs(d_prj / d_gt)
            if (0.8 < dif < 1.25):
                kpt0_valid.append(True)
            else:
                # print("WARNING DIFF: ", dif, d_gt)
                kpt0_valid.append(False)
        else:
            kpt0_valid.append(False)

    if kpt0_valid.count(True) == 0:
        # print('BAD Overlap !')
        return 0,0,0

    norm_cross = np.sqrt(2 - 2 * np.clip(np.dot(desc0.T, desc1), -1, 1))
    match_id = linear_sum_assignment(cost_matrix=(norm_cross))
    match_score = norm_cross[match_id]
    matches = np.stack((match_id[0], match_id[1], match_score), axis=1)

    kpt1 = kpts1.astype(int)
    thresh = 0.9
    kp0_gt = kpt0_gt
    kp0_valid = kpt0_valid

    match_state = []
    for id0, id1, mVal in matches:
        # p0 = kpt0[int(id0)]
        p0_gt = kp0_gt[int(id0)]
        valid = kp0_valid[int(id0)]
        p1 = kpt1[int(id1)]

        dis = np.linalg.norm(p0_gt - p1)
        if (mVal < thresh):
            if (dis < 8):
                match_state.append('TRUE')
            else:
                match_state.append('FALSE')
        else:
            if valid:
                match_state.append('SKIP_BUT_VALID')
            else:
                match_state.append('SKIP')

    precision, recall, f1 = 0, 0, 0
    if (match_state.count('TRUE') > 0):
        precision = match_state.count('TRUE') / (match_state.count('TRUE') + match_state.count('FALSE'))
        recall = match_state.count('TRUE') / (match_state.count('TRUE') + match_state.count('SKIP_BUT_VALID'))
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1

def getMatchPrecision_NN2way(source_id,target_id):
    source = getFrameInfo(source_id)
    target = getFrameInfo(target_id)
    source_cam = PinholeCameraCal3_S2(Pose3(source['transform']), K)
    target_cam = PinholeCameraCal3_S2(Pose3(target['transform']), K)

    src_gray = cv.cvtColor(source['color'], cv.COLOR_RGB2GRAY)
    src_p3d = getPoint3D(source)
    tar_gray = cv.cvtColor(target['color'], cv.COLOR_RGB2GRAY)

    # FRAME 0
    frame_tensor = frame2tensor(src_gray, dnn_device)
    keys = ['keypoints', 'scores', 'descriptors']
    last_data = matching.superpoint({'image': frame_tensor})
    last_data = {k + '0': last_data[k] for k in keys}
    last_data['image0'] = frame_tensor
    last_frame = src_gray

    # FRAME 1
    frame_tensor_tar = frame2tensor(tar_gray, dnn_device)
    current_data = matching.superpoint({'image': frame_tensor_tar})
    current_data = {k + '1': current_data[k] for k in keys}
    current_data['image1'] = frame_tensor_tar

    kpts0 = last_data['keypoints0'][0].cpu().numpy()
    kpts1 = current_data['keypoints1'][0].cpu().numpy()
    desc0 = last_data['descriptors0'][0].cpu().numpy()
    desc1 = current_data['descriptors1'][0].cpu().numpy()

    kpt0_gt = []
    kpt0_valid = []

    for kp in kpts0:
        x, y = kp
        p0w = Point3(src_p3d[int(y * 640 + x)])
        p0_target, d_prj = cal_Point_Project_TartanCam(target_cam, p0w)
        kpt0_gt.append(p0_target)
        if (0 < p0_target[0] < 640) & (0 < p0_target[1] < 480):
            d_gt = target['depth'][p0_target[1], p0_target[0]]
            dif = abs(d_prj / d_gt)
            if (0.8 < dif < 1.25):
                kpt0_valid.append(True)
            else:
                # print("WARNING DIFF: ", dif, d_gt)
                kpt0_valid.append(False)
        else:
            kpt0_valid.append(False)

    if kpt0_valid.count(True) == 0:
        # print('BAD Overlap !')
        return 0,0,0

    dmat = np.dot(desc0.T, desc1)
    dmat = np.sqrt(2 - 2 * np.clip(dmat, -1, 1))

    idx = np.argmin(dmat, axis=1)

    scores = dmat[np.arange(dmat.shape[0]), idx]
    nn_thresh = 1.0
    keep = scores < nn_thresh

    idx2 = np.argmin(dmat, axis=0)
    keep_bi = np.arange(len(idx)) == idx2[idx]
    keep = np.logical_and(keep, keep_bi)

    for i in range(len(idx)):
        if keep[i] == False:
            idx[i] = -1

    match_state = []
    matches = idx

    for id0, id1 in enumerate(matches):
        p0_gt = kpt0_gt[int(id0)]
        valid = kpt0_valid[int(id0)]

        if (id1 > -1):
            p1 = kpts1[int(id1)]
            dis = np.linalg.norm(p0_gt - p1)
            if (dis < 8):
                match_state.append('TRUE')
            else:
                match_state.append('FALSE')
        else:
            if valid:
                match_state.append('SKIP_BUT_VALID')
            else:
                match_state.append('SKIP')

    precision = 0
    recall = 0
    f1 = 0
    if (match_state.count('TRUE') > 0):
        precision = match_state.count('TRUE') / (match_state.count('TRUE') + match_state.count('FALSE'))
        recall = match_state.count('TRUE') / (match_state.count('TRUE') + match_state.count('SKIP_BUT_VALID'))
        f1 = 2 * precision * recall / (precision + recall)

    # print('Precision = ', precision, '; Recall = ', recall, '; F1 = ', f1)
    return precision, recall, f1

def traceSquence(sce,lvl,traj):
    path = os.path.join(rootDIR, sce, lvl, traj, '')
    # print(path)
    print_name = sce + '_' + lvl + '_' + traj

    step = 2
    src_id = 0
    last_id = len(files_rgb_left)
    total_rs = []

    while (src_id + 4) < last_id:
        # print(src_id)
        # p4, r4, f4 = getMatchPrecision_NN2way(src_id, src_id + 4)
        p4, r4, f4 = getMatchPrecision_Hungarian(src_id, src_id + 4)
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
est_rs = {}
names = []
f1 = []
for sce in keys:
    for lv in levels:
        trajs = db[sce][lv]
        for traj in trajs:
            path = os.path.join(rootDIR, sce, lv, traj, '')
            files_rgb_left, files_rgb_right, files_depth_left, poses_quad = getDataLists(dir=path, skip=5)
            poses_mat44 = pos_quats2SE_matrices(poses_quad)
            traceSquence(sce,lv,traj)
    print('----------')

# for traj in db['office']['Hard']:
#     path = os.path.join(rootDIR,'office','Hard', traj, '')
#     files_rgb_left, files_rgb_right, files_depth_left, poses_quad = getDataLists(dir=path, skip=5)
#     poses_mat44 = pos_quats2SE_matrices(poses_quad)
#
#     step = 2
#     src_id = 0
#     last_id = len(files_rgb_left)
#
#     total_rs = []
#
#     while (src_id + 4) < last_id:
#         p4, r4, f4 = getMatchPrecision_NN2way(src_id, src_id + 4)
#         rs = [p4, r4, f4]
#         total_rs.append(rs)
#         src_id += step
#
#     rs_np = np.asarray(total_rs)
#     p4 = rs_np[:, 0]
#     r4 = rs_np[:, 1]
#     f4 = rs_np[:, 2]
#
#     print(path)
#     print('Match 20-step:',  '%6f' % p4[p4 != 0].mean(), '%6f' % r4[r4 != 0].mean(), '%6f' % f4[f4 != 0].mean())

# source = getFrameInfo(32)
# target = getFrameInfo(36)
# source_cam = PinholeCameraCal3_S2(Pose3(source['transform']), K)
# target_cam = PinholeCameraCal3_S2(Pose3(target['transform']), K)
# src_gray = cv.cvtColor(source['color'], cv.COLOR_RGB2GRAY)
# src_p3d = getPoint3D(source)
# tar_gray = cv.cvtColor(target['color'], cv.COLOR_RGB2GRAY)
# # FRAME 0
# frame_tensor = frame2tensor(src_gray, dnn_device)
# keys = ['keypoints', 'scores', 'descriptors']
# last_data = matching.superpoint({'image': frame_tensor})
# last_data = {k + '0': last_data[k] for k in keys}
# last_data['image0'] = frame_tensor
# last_frame = src_gray
# # FRAME 1
# frame_tensor_tar = frame2tensor(tar_gray, dnn_device)
# current_data = matching.superpoint({'image': frame_tensor_tar})
# current_data = {k + '1': current_data[k] for k in keys}
# current_data['image1'] = frame_tensor_tar
#
# kpts0 = last_data['keypoints0'][0].cpu().numpy()
# kpts1 = current_data['keypoints1'][0].cpu().numpy()
# desc0 = last_data['descriptors0'][0].cpu().numpy()
# desc1 = current_data['descriptors1'][0].cpu().numpy()
#
# kpt0_gt = []
# kpt0_valid = []
#
# for kp in kpts0:
#     x, y = kp
#     p0w = Point3(src_p3d[int(y * 640 + x)])
#     p0_target, d_prj = cal_Point_Project_TartanCam(target_cam, p0w)
#     kpt0_gt.append(p0_target)
#     if (0 < p0_target[0] < 640) & (0 < p0_target[1] < 480):
#         d_gt = target['depth'][p0_target[1], p0_target[0]]
#         dif = abs(d_prj / d_gt)
#         if (0.8 < dif < 1.25):
#             kpt0_valid.append(True)
#         else:
#             # print("WARNING DIFF: ", dif, d_gt)
#             kpt0_valid.append(False)
#     else:
#         kpt0_valid.append(False)
#
# norm_cross = np.sqrt(2 - 2 * np.clip(np.dot(desc0.T, desc1), -1, 1))
# match_id = linear_sum_assignment(cost_matrix=(norm_cross))
# match_score = norm_cross[match_id]
# matches = np.stack((match_id[0], match_id[1], match_score), axis=1)

# dmat = np.dot(desc0.T, desc1)
# dmat = np.sqrt(2 - 2 * np.clip(dmat, -1, 1))
# idx = np.argmin(dmat, axis=1)
# scores = dmat[np.arange(dmat.shape[0]), idx]
# nn_thresh = 1.0
# keep = scores < nn_thresh
#
# idx2 = np.argmin(dmat, axis=0)
# keep_bi = np.arange(len(idx)) == idx2[idx]
# keep = np.logical_and(keep, keep_bi)
# for i in range(len(idx)):
#     if keep[i] == False:
#         idx[i] = -1
#
# kpt0_gt = []
# kpt0_valid = []
# for kp in kpts0:
#     x, y = kp
#     p0w = Point3(src_p3d[int(y * 640 + x)])
#     p0_target, d_prj = cal_Point_Project_TartanCam(target_cam, p0w)
#     kpt0_gt.append(p0_target)
#     if (0 < p0_target[0] < 640) & (0 < p0_target[1] < 480):
#         d_gt = target['depth'][p0_target[1], p0_target[0]]
#         dif = abs(d_prj / d_gt)
#         if (0.8 < dif < 1.25):
#             kpt0_valid.append(True)
#         else:
#             # print("WARNING DIFF: ", dif, d_gt)
#             kpt0_valid.append(False)
#     else:
#         kpt0_valid.append(False)
#
# match_state = []
# matches = idx
# for id0, id1 in enumerate(matches):
#     p0_gt = kpt0_gt[int(id0)]
#     valid = kpt0_valid[int(id0)]
#
#     if (id1 > -1):
#         p1 = kpts1[int(id1)]
#         dis = np.linalg.norm(p0_gt - p1)
#         if (dis < 8):
#             match_state.append('TRUE')
#         else:
#             match_state.append('FALSE')
#     else:
#         if valid:
#             match_state.append('SKIP_BUT_VALID')
#         else:
#             match_state.append('SKIP')
#
# precision = 0
# recall = 0
# f1 = 0
# if (match_state.count('TRUE') > 0):
#     precision = match_state.count('TRUE') / (match_state.count('TRUE') + match_state.count('FALSE'))
#     recall = match_state.count('TRUE') / (match_state.count('TRUE') + match_state.count('SKIP_BUT_VALID'))
#     f1 = 2 * precision * recall / (precision + recall)
#
# print('Precision = ', precision, '; Recall = ', recall,  '; F1 = ', f1)
