import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from skimage import io
import cv2 as cv
# import pyelsed
import ot

from scipy.optimize import linear_sum_assignment

from config import camera, RGBDImage, PointCloud, TriangleMesh, Image, Vector3dVector, draw_geometries
from config import gtsam, Point2, Point3, Rot3, Pose3, Cal3_S2, Values, PinholeCameraCal3_S2

from SlamUtils.Loader.TartanAir import getDataSequences, getDataLists, reload_with_MSCKF_Estimation, tartan_camExtr
from SlamUtils.transformation import pos_quats2SEs, pos_quats2SE_matrices, pose2motion, SEs2ses, line2mat, tartan2kitti

from Semantics.image_proc_3D import generate3D, getPointsProject2D, \
    cal_Point_Project_General, cal_Point_Project_TartanCam, \
    getDisparityTartanAIR, get3DLocalFromDisparity
from Semantics.image_proc_2D import getSuperPoints_v2, hungarianMatch, sinkhornMatch, \
    getImgSobel,getSobelMask,showRGB, showDepth, showNorm

path = getDataSequences(scenario='office', level='Easy', seq_num=4)
# path = getDataSequences(scenario = 'seasidetown', level='Easy', seq_num=0)
# path = getDataSequences(scenario = 'neighborhood', level='Easy', seq_num=0)
files_rgb_left, files_rgb_right, files_depth_left, poselist = getDataLists(dir=path)
# poses_mat44 = pos_quats2SE_matrices(poselist)

pose_est_mat44, pose_gt_mat44, \
files_rgb_left, files_rgb_right, \
files_depth_left = reload_with_MSCKF_Estimation(files_rgb_left,files_rgb_right,files_depth_left, seq_name = 'office_004')
# pose_est_mat44, pose_gt_mat44, \
# files_rgb_left, files_rgb_right, \
# files_depth_left = reload_with_MSCKF_Estimation(files_rgb_left,files_rgb_right,files_depth_left, seq_name = 'neighborhood_000')

config_cam = {'width':640, 'height':480, 'fx':320, 'fy':320, 'cx':320, 'cy':240}
camIntr = camera.PinholeCameraIntrinsic(**config_cam) #open3d if need
camExtr = tartan_camExtr # World to cam
inv_camExtr = np.linalg.inv(camExtr) # Cam to World
K = Cal3_S2(320, 320, 0.0, 320, 240)

def getFrameInfo(id):
    frame = {   'color': io.imread(files_rgb_left[id]),
                'color_right': io.imread(files_rgb_right[id]),
                'depth': np.load(files_depth_left[id]),
                'transform': pose_gt_mat44[id],
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
    print('Precision = ', precision, '; Recall = ', recall, '; F1 = ', f1)
    return precision, recall, f1, pts0['pts'].shape[0]

def getMatchPrecision(src_id, tar_id):
    source = getFrameInfo(src_id) #(30 * 5)  # 32*5
    target = getFrameInfo(tar_id) #(34 * 5)  # 36*5
    motion = np.linalg.inv(source['transform']).dot(target['transform'])

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
    # src_disparity = 80./source['depth']

    # showDepth(src_disparity)

    kp0 = pts0['pts']
    kp1 = pts1['pts']
    kp0_3d = get3DLocalFromDisparity(kp0, src_disparity)

    num_detected = pts0['pts'].shape[0]
    kp0_gt, kp0_valid = getPointsProject2D(pts_2d=pts0['pts'], pts_3d=source['point3D'], target_cam=target_cam)
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

    import time
    st = time.time()
    Gs = ot.sinkhorn(a, b, norm_cross, reg=1e-1, numItermax=20)
    sktime = time.time()

    ### Sinkhorn with 3D project map
    match_id = linear_sum_assignment(cost_matrix = (1-Gs)+map)
    et = time.time()
    match_score = norm_cross[match_id]
    matches = np.stack((match_id[0], match_id[1], match_score), axis=1)

    ### Hungarian with 3D map
    # match_id = linear_sum_assignment(cost_matrix = norm_cross+map)
    # match_score = norm_cross[match_id]
    # matches = np.stack((match_id[0], match_id[1], match_score), axis=1)
    print(src_id, ' -> ', tar_id, ': ', pts0['pts'].shape[0])

    # print('Match Execution time:', et - st, 'seconds')
    # print('Sinkhorn:', sktime - st, 'seconds')
    # print('Hungarian:', et - sktime, 'seconds')
    # return et - st, sktime - st, et - sktime
    # return evalScores(pts0, pts1, matches, kp0_gt, kp0_valid, thresh=.7) # for improve Hungarian precision

    # Motion check ...
    motion = Pose3(motion)
    rpy = motion.rotation().rpy()
    xyz = motion.translation()
    r_cost = np.linalg.norm(rpy)
    t_cost = np.linalg.norm(xyz)
    print('Transform:', r_cost, t_cost)
    if ((r_cost < 0.25) & (t_cost < 1.2)):
        return evalScores(pts0, pts1, matches, kp0_gt, kp0_valid, thresh=1.1)

    return evalScores(pts0, pts1, matches, kp0_gt, kp0_valid, thresh=1.2)

step = 10
src_id = 0
last_id = len(files_rgb_left)
total_rs = []

while (src_id+20)<last_id:
    # p1,r1,f1 = getMatchPrecision(src_id,src_id + 5)
    # p2,r2,f2 = getMatchPrecision(src_id, src_id + 10)
    p4,r4,f4,kp4 = getMatchPrecision(src_id, src_id + 20)

    # rs = [p1,r1,f1,p2,r2,f2,p4,r4,f4]
    # total_rs.append(rs)

    rs = [p4,r4,f4,kp4]
    total_rs.append(rs)
    src_id+=step

# np.savetxt(fname = 'office_004_hg3d.txt', X=np.asarray(total_rs), header='p1 r1 p2 r2 p4 r4 #steps=[5_10_20]')
# np.savetxt(fname = 'office_004_sh3d.txt', X=np.asarray(total_rs), header='p1 r1 p2 r2 p4 r4 #steps=[5_10_20]')
# np.savetxt(fname = 'neighborhood_000_hg3d.txt', X=np.asarray(total_rs), header='p1 r1 p2 r2 p4 r4 #steps=[5_10_20]')
# np.savetxt(fname = 'neighborhood_000_sh3d.txt', X=np.asarray(total_rs), header='p1 r1 p2 r2 p4 r4 #steps=[5_10_20]')

rs_np = np.asarray(total_rs)

p = rs_np[:,0];    r = rs_np[:,1];    f = rs_np[:,2];   kp = rs_np[:,3];
print('Match 20-step: ', p[p!=0].mean(), r[r!=0].mean(), f[f!=0].mean())

# print('Average: ', rs_np[:,0].mean(), rs_np[:,1].mean(), rs_np[:,2].mean())
# print('Sinkhorn: ', )

# p1 = rs_np[:,0];    r1 = rs_np[:,1];    f1 = rs_np[:,2]
# p2 = rs_np[:,3];    r2 = rs_np[:,4];    f2 = rs_np[:,5]
# p4 = rs_np[:,6];    r4 = rs_np[:,7];    f4 = rs_np[:,8]
#
# print('Match 05-step: ', p1[p1!=0].mean(), r1[r1!=0].mean(), f1[f1!=0].mean())
# print('Match 10-step: ', p2[p2!=0].mean(), r2[r2!=0].mean(), f2[f2!=0].mean())
# print('Match 20-step: ', p4[p4!=0].mean(), r4[r4!=0].mean(), f4[f4!=0].mean())

# print('Match 05-step: ', p1[p1!=0][:100].mean(), r1[r1!=0][:100].mean(), f1[f1!=0][:100].mean())
# print('Match 10-step: ', p2[p2!=0][:100].mean(), r2[r2!=0][:100].mean(), f2[f2!=0][:100].mean())
# print('Match 20-step: ', p4[p4!=0][:100].mean(), r4[r4!=0][:100].mean(), f4[f4!=0][:100].mean())



# source = getFrameInfo(30*5) # 32*5
# target = getFrameInfo(34*5) # 36*5
# motion = np.linalg.inv(source['transform']).dot(target['transform'])
# # motion_gt = np.linalg.inv(pose_gt_mat44[30*5]).dot(pose_gt_mat44[34*5])
#
# generate3D(source)
# src_p3d = source['point3D']
# target_cam = PinholeCameraCal3_S2(Pose3(target['transform']), K)
#
# pts0 = getSuperPoints_v2(cv.cvtColor(source['color'], cv.COLOR_RGB2GRAY))
# pts1 = getSuperPoints_v2(cv.cvtColor(target['color'], cv.COLOR_RGB2GRAY))
# desc0 = pts0['desc']
# desc1 = pts1['desc']
#
# norm_self_src = np.sqrt(2 - 2 * np.clip(np.dot(desc0.T, desc0), -1, 1))
# norm_self_tar = np.sqrt(2 - 2 * np.clip(np.dot(desc1.T, desc1), -1, 1))
# norm_cross = np.sqrt(2 - 2 * np.clip(np.dot(desc0.T, desc1), -1, 1))
#
# # get stereo map
# src_disparity = getDisparityTartanAIR(source['color'], source['color_right'])
# # showDepth(src_disparity)
#
# kp0 = pts0['pts']
# kp1 = pts1['pts']
# kp0_3d = get3DLocalFromDisparity(kp0, src_disparity)
#
# num_detected = pts0['pts'].shape[0]
# kp0_gt, kp0_valid = getPointsProject2D(pts_2d=pts0['pts'], pts_3d=source['point3D'], target_cam=target_cam)
# cam_target_predicted = PinholeCameraCal3_S2(Pose3(motion), K)
#
# extr = camExtr[:3,:3] # for convert global point to cam points  -> z-forward
# inv_extr = inv_camExtr[:3,:3] # for convert cam (z-forward) to NED -> z-down
#
# kp0_expect = []
# for pt3d in kp0_3d:
#     pt2d = cal_Point_Project_General(cam_target_predicted, pt3d, extr)
#     kp0_expect.append(pt2d)
# kp0_expect = np.asarray(kp0_expect)
#
# #gen distance-table
# map = np.zeros(shape=[kp0.shape[0], kp1.shape[0]])
# std = 64
# for i,p0_e in enumerate(kp0_expect):
#     for j,p1 in enumerate(kp1):
#         map[i,j] = np.sqrt(np.linalg.norm(p0_e - p1)/std)
#
# ### Get Sinkhorn Map
# a, b = [], []
# for pt_his in norm_self_src:
#     a.append(pt_his.mean() - 1)
# for pt_his in norm_self_tar:
#     b.append(pt_his.mean() - 1)
# Gs = ot.sinkhorn(a, b, norm_cross, reg=1e-1, numItermax=20)
#
# ### Simple Sinkhorn
# # matches = hungarianMatch(norm_cross = (1-Gs))
#
# ### Sinkhorn with 3D project map
# match_id = linear_sum_assignment(cost_matrix = (1-Gs)+map)
# match_score = norm_cross[match_id]
# matches = np.stack((match_id[0], match_id[1], match_score), axis=1)
#
# ### Hungarian with 3D map
# # match_id = linear_sum_assignment(cost_matrix = norm_cross+map)
# # match_score = norm_cross[match_id]
# # matches = np.stack((match_id[0], match_id[1], match_score), axis=1)
#
#
# # Hungarian_Thresh = 0.7
# # Sinkhorn_Thresh = 0.75 -> best precision; 0.95 best F1; 1.0 best Recall
# # Hungarian_3D = 0.9
# # Sinkhorn_3D = 0.9
#
# evalScores(pts0,pts1,matches,kp0_gt,kp0_valid,thresh=.9)

# 32-> 36 best:
# Sinkhorn only: Precision =  0.9746835443037974 ; Recall =  0.27898550724637683 ; F1 =  0.4338028169014085
# Sinkhorn + Map: Precision =  0.9541984732824428 ; Recall =  0.8680555555555556 ; F1 =  0.9090909090909092
# Hungarian + Map:
# + thesh=0.7 => best precision -> 0.9652173913043478 ; Recall =  0.7602739726027398 ; F1 =  0.8505747126436782
# + thres=0.9 => best F1 score -> 0.9253731343283582 ; Recall =  0.8701754385964913 ; F1 =  0.8969258589511755

# SOTA SuperGlue: Precision =  0.991869918699187 ; Recall =  0.8271186440677966 ; F1 =  0.9020332717190387


# retval, rvec, tvec, inliers = cv.solvePnPRansac(kp0_3d_local[m0],kp1[m1],camIntr.intrinsic_matrix, np.zeros(5))