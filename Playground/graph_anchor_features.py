import numpy as np
import sys
sys.path.append('../')
import numpy as np
import cv2 as cv
import ot
from skimage import io

from config import camera, RGBDImage, PointCloud, TriangleMesh, Image, Vector3dVector, draw_geometries
from config import gtsam, Point2, Point3, Rot3, Pose3, Cal3_S2, Values, PinholeCameraCal3_S2

from Semantics.image_proc_2D import getImgSobel,getSobelMask, getLineDistance, getLineMinMax,\
                                    showRGB, showDepth, showNorm, getSuperPoints_v2, getAnchorPoints
from Semantics.image_proc_3D import cal_Point_Project_TartanCam, generate3D, \
    getDisparityTartanAIR, get3DLocalFromDisparity, getGroundTruth

from SlamUtils.transformation import pos_quats2SEs, pos_quats2SE_matrices, pose2motion, SEs2ses, line2mat, tartan2kitti
from SlamUtils.visualization import getVisualizationBB, getKeyframe
from SlamUtils.Loader.TartanAir import getRootDir, getDataSequences, getDataLists, tartan_camExtr

from scipy.optimize import linear_sum_assignment

rootDIR = getRootDir()
path = getDataSequences(root=rootDIR, scenario='office', level='Easy', seq_num=4)
# path = getDataSequences(root=rootDIR, scenario='neighborhood', level='Easy', seq_num=0) # 104 -> 108 & SKIP 5
files_rgb_left, files_rgb_right, files_depth_left, poselist = getDataLists(dir=path, skip=1)
poses_mat44 = pos_quats2SE_matrices(poselist)

config_cam = {'width':640, 'height':480, 'fx':320, 'fy':320, 'cx':320, 'cy':240}
camIntr = camera.PinholeCameraIntrinsic(**config_cam) #open3d if need
camExtr = tartan_camExtr

K = Cal3_S2(320, 320, 0.0, 320, 240)

### ----- DATA ----- ###

def getFrameInfo(id):
    frame = {
        'color': io.imread(files_rgb_left[id]),
        'color_right': io.imread(files_rgb_right[id]),
        'depth': np.load(files_depth_left[id]),
        'transform': poses_mat44[id],
        'intr': camIntr,
        'extr': camExtr
    }
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
    print('Precision = ', precision, '; Recall = ', recall, '; F1 = ', f1, '; Total = ', len(match_state))
    return precision, recall, f1

def getMatchPrecision(src_id, tar_id):
    print(src_id, ' -> ', tar_id)
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
    # src_disparity = 80./source['depth']
    # tar_disparity = 80./target['depth']

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

    map = np.zeros(shape=[kp0.shape[0], kp1.shape[0]])
    std = 64
    for i, p0_e in enumerate(kp0_expected):
        for j, p1 in enumerate(kp1):
            map[i, j] = np.sqrt(np.linalg.norm(p0_e - p1) / std)

    match_id = linear_sum_assignment(cost_matrix=(1 - Gs) + map)
    match_score = norm_cross[match_id]
    matches = np.stack((match_id[0], match_id[1], match_score), axis=1)
    return evalScores(pts0, pts1, matches, kpt0_gt, kpt0_valid, thresh=1.2)

step = 10
src_id = 0
last_id = len(files_rgb_left)
total_rs = []

while (src_id+20)<last_id:
    # p1,r1,f1 = getMatchPrecision(src_id,src_id + 5)
    # p2,r2,f2 = getMatchPrecision(src_id, src_id + 10)
    # p4,r4,f4 = getMatchPrecision(src_id, src_id + 20)
    # rs = [p1,r1,f1,p2,r2,f2,p4,r4,f4]

    p1, r1, f1 = getMatchPrecision(src_id, src_id + 20)
    rs = [p1, r1, f1]
    total_rs.append(rs)
    src_id+=step

rs_np = np.asarray(total_rs)

# p1 = rs_np[:,0];    r1 = rs_np[:,1];    f1 = rs_np[:,2]
# p2 = rs_np[:,3];    r2 = rs_np[:,4];    f2 = rs_np[:,5]
# p4 = rs_np[:,6];    r4 = rs_np[:,7];    f4 = rs_np[:,8]
#
# print('Match 05-step: ', p1[p1!=0].mean(), r1[r1!=0].mean(), f1[f1!=0].mean())
# print('Match 10-step: ', p2[p2!=0].mean(), r2[r2!=0].mean(), f2[f2!=0].mean())
# print('Match 20-step: ', p4[p4!=0].mean(), r4[r4!=0].mean(), f4[f4!=0].mean())

p1 = rs_np[:,0];    r1 = rs_np[:,1];    f1 = rs_np[:,2]
print('Match 20-step: ', p1[p1!=0].mean(), r1[r1!=0].mean(), f1[f1!=0].mean())

source = getFrameInfo(160)
target = getFrameInfo(180)
generate3D(source)
src_p3d = source['point3D']

source_cam = PinholeCameraCal3_S2(Pose3(source['transform']),K)
target_cam = PinholeCameraCal3_S2(Pose3(target['transform']),K)

src_gray = cv.cvtColor(source['color'], cv.COLOR_RGB2GRAY)
tar_gray = cv.cvtColor(target['color'], cv.COLOR_RGB2GRAY)
src_disparity = getDisparityTartanAIR(source['color'], source['color_right'])
tar_disparity = getDisparityTartanAIR(target['color'], target['color_right'])
# src_disparity = 80./source['depth']
# tar_disparity = 80./target['depth']

pts0 = getSuperPoints_v2(src_gray)
pts1 = getSuperPoints_v2(tar_gray)

kp0 = pts0['pts']
kp1 = pts1['pts']
desc0 = pts0['desc']
desc1 = pts1['desc']

norm_self_src = np.sqrt(2 - 2 * np.clip(np.dot(desc0.T, desc0), -1, 1))
norm_self_tar = np.sqrt(2 - 2 * np.clip(np.dot(desc1.T, desc1), -1, 1))
his_src = np.mean(norm_self_src,axis=0)/2
his_tar = np.mean(norm_self_tar,axis=0)/2

norm_cross = np.sqrt(2 - 2 * np.clip(np.dot(desc0.T, desc1), -1, 1))
kp0_3d = get3DLocalFromDisparity(kp0, src_disparity)
kp1_3d = get3DLocalFromDisparity(kp1, tar_disparity)

# Bench ...
kpt0_gt, kpt0_valid = getGroundTruth(src_p3d_world = src_p3d,
                                     src_p2d = kp0,
                                     target_cam = target_cam)

# Cross check = BAD -> Sinkhorn CROSS CHECK
matches,Gs = getAnchorPoints(norm_cross, his_src, his_tar, sorting=True)
level = matches.shape[0]/8

# evalScores(pts0, pts1, matches, kpt0_gt, kpt0_valid, thresh=.9)
anchor_src_id = matches.T[0].astype(int)
anchor_tar_id = matches.T[1].astype(int)

src_anchor = kp0[anchor_src_id]
src_anchor_3D = kp0_3d[anchor_src_id]
tar_anchor = kp1[anchor_tar_id]
tar_anchor_3D = kp1_3d[anchor_tar_id]

dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
s,r,t,x = cv.solvePnPRansac(objectPoints=src_anchor_3D, imagePoints=tar_anchor, cameraMatrix=camIntr.intrinsic_matrix, distCoeffs=None)
kp0_expected, jacobian = cv.projectPoints(kp0_3d, r, t, camIntr.intrinsic_matrix, dist_coeffs)

map = np.zeros(shape=[kp0.shape[0], kp1.shape[0]])
std = 16
for i, p0_e in enumerate(kp0_expected):
    for j, p1 in enumerate(kp1):
        map[i, j] = np.sqrt(np.linalg.norm(p0_e - p1) / std)

match_id = linear_sum_assignment(cost_matrix = (1-Gs)+map)
match_score = norm_cross[match_id]
matches = np.stack((match_id[0], match_id[1], match_score), axis=1)

evalScores(pts0, pts1, matches, kpt0_gt, kpt0_valid, thresh=.9)

# debug_view_src = source['color'].copy()
# debug_view_tar = target['color'].copy()
# for pt in kp0:
#     cv.drawMarker(debug_view_src, pt.astype(np.int32), (255,0,0), markerType=cv.MARKER_CROSS, markerSize=7, thickness=2)
#
# for pt in kp0_expected:
#     cv.drawMarker(debug_view_tar, pt[0].astype(np.int32), (0,0,255), markerType=cv.MARKER_CROSS, markerSize=7, thickness=2)
#
# for pt in kp1:
#     cv.drawMarker(debug_view_tar, pt.astype(np.int32), (255,0,0), markerType=cv.MARKER_CROSS, markerSize=7, thickness=1)

# log(dbase)*(1/log(d1)-1/log(d2))
# Generate External G_descriptors

base_cm = 25
log_base = np.log(base_cm) # 10cm
gDesc0 = np.zeros((matches.shape[0],desc0.shape[1]))
gDesc1 = np.zeros((matches.shape[0],desc1.shape[1]))

for id0, p2d in enumerate(kp0):
    p3d = kp0_3d[id0]
    d_cm = np.linalg.norm(src_anchor_3D-p3d,axis=1)*100
    for zero_id in np.where(d_cm<base_cm)[0]:
        d_cm[zero_id] = base_cm
    gDesc0[:,id0] = log_base/np.log(d_cm)

for id1, p2d in enumerate(kp1):
    p3d = kp1_3d[id1]
    d_cm = np.linalg.norm(src_anchor_3D-p3d,axis=1)*100
    for zero_id in np.where(d_cm<base_cm)[0]:
        d_cm[zero_id] = base_cm
    gDesc1[:,id1] = log_base/np.log(d_cm)

norm_cross_g = np.zeros(norm_cross.shape)
for i in range(gDesc0.shape[1]) :
    src_gDesc = gDesc0[:,i]
    for j in range(gDesc1.shape[1]):
        tar_gDesc = gDesc1[:,j]
        norm_cross_g[i,j] = np.linalg.norm(src_gDesc-tar_gDesc)


# for id0, p2d in enumerate(kp0):
#     p3d = kp0_3d[id0]
#     d_cm = np.linalg.norm(src_anchor_3D-p3d,axis=1)*100
#     for zero_id in np.where(d_cm<base_cm)[0]:
#         d_cm[zero_id] = base_cm
#     gDesc0[:,id0] = base_cm/d_cm
#
# for id1, p2d in enumerate(kp1):
#     p3d = kp1_3d[id1]
#     d_cm = np.linalg.norm(src_anchor_3D-p3d,axis=1)*100
#     for zero_id in np.where(d_cm<base_cm)[0]:
#         d_cm[zero_id] = base_cm
#     gDesc1[:,id1] = base_cm/d_cm
#
# norm_cross_g = np.sqrt(2 - 2 * np.clip(np.dot(gDesc0.T, gDesc1), -1, 1))

magic = norm_cross_g + norm_cross

match_id = linear_sum_assignment(cost_matrix=magic, maximize=False)
match_score = norm_cross[match_id]
matches_xxx = np.stack((match_id[0], match_id[1], match_score), axis=1)
evalScores(pts0, pts1, matches_xxx, kpt0_gt, kpt0_valid, thresh=.9)

# np.linalg.norm(gDesc0[:,301] - gDesc1[:,285])

match_id = linear_sum_assignment(cost_matrix=norm_cross, maximize=False)
match_score = norm_cross[match_id]
matches_noob = np.stack((match_id[0], match_id[1], match_score), axis=1)
evalScores(pts0, pts1, matches_noob, kpt0_gt, kpt0_valid, thresh=.9)