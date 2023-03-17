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

# check overlapping
def checkLinesOverlap(s0,s1):
    p1 = s0[0:2]; p2 = s0[2:4]
    p3 = s1[0:2]; p4 = s1[2:4]
    d1 = np.abs(np.cross(p2-p1, p1-p3)) / np.linalg.norm(p2-p1)
    d2 = np.abs(np.cross(p2-p1, p1-p4)) / np.linalg.norm(p2-p1)
    if (d1<5) & (d2<5):
        return False;
    return True

def getLines(gray_img):
    src_gray_adjusted = cv.equalizeHist(gray_img)
    segments, scores = pyelsed.detect(src_gray_adjusted,
                                      gradientThreshold=30,
                                      minLineLen=15,
                                      lineFitErrThreshold=0.2,
                                      pxToSegmentDistTh=1.5,
                                      validationTh=0.15)
    id = np.argsort(scores)[::-1]
    segments = segments[id]
    scores = scores[id]

    num_lines = scores.shape[0]
    overlap_id = np.full(scores.shape, True)
    for st_id in range(num_lines - 2):
        s_ref = segments[st_id]

        for check_id in range(st_id + 1, num_lines):
            if (overlap_id[check_id]):
                s_check = segments[check_id]
                overlap_id[check_id] = checkLinesOverlap(s_ref, s_check)

    segments = segments[overlap_id]
    scores = scores[overlap_id]
    return segments,scores

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

def checkAnchorScores(src_id, tar_id):
    source = getFrameInfo(src_id)
    target = getFrameInfo(tar_id)
    generate3D(source)
    src_p3d = source['point3D']

    source_cam = PinholeCameraCal3_S2(Pose3(source['transform']),K)
    target_cam = PinholeCameraCal3_S2(Pose3(target['transform']),K)

    src_gray = cv.cvtColor(source['color'], cv.COLOR_RGB2GRAY)
    tar_gray = cv.cvtColor(target['color'], cv.COLOR_RGB2GRAY)
    # src_disparity = getDisparityTartanAIR(source['color'], source['color_right'])
    # tar_disparity = getDisparityTartanAIR(target['color'], target['color_right'])

    # src_disparity = 80. / source['depth']
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
    # kp0_3d = get3DLocalFromDisparity(kp0, src_disparity)
    # kp1_3d = get3DLocalFromDisparity(kp1, tar_disparity)

    # Bench ...
    kpt0_gt, kpt0_valid = getGroundTruth(src_p3d_world = src_p3d,
                                         src_p2d = kp0,
                                         target_cam = target_cam)

    # Cross check = BAD -> Sinkhorn CROSS CHECK
    matches = getAnchorPoints(norm_cross, his_src, his_tar, sort_id=False)
    evalScores(pts0, pts1, matches, kpt0_gt, kpt0_valid, thresh=1.0)

...

step = 10
src_id = 0
last_id = len(files_rgb_left)
total_rs = []

while (src_id+20)<last_id:
    checkAnchorScores(src_id, src_id + 20)
    src_id+=step


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

# src_disparity = 80. / source['depth']
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
matches = getAnchorPoints(norm_cross, his_src, his_tar, sort_id=False)
evalScores(pts0, pts1, matches, kpt0_gt, kpt0_valid, thresh=.9)

# MAGICS:
# A. Get Anchor points - KNN (?)

# B. Get ...

# C. Get ...

# 2. Line Descriptors:
import pyelsed
segments,scores = getLines(src_gray)
segment_supports = []
radius_soft_thresh = 64
for s in segments:
    p0 = Point2(s[0], s[1])
    p1 = Point2(s[2], s[3])
    min_x,min_y,max_x,max_y = getLineMinMax(s,radius_soft_thresh)

    sub_graph_id = []
    for id,kp in enumerate(kp0):
        dis_line = getLineDistance(p0, p1, kp)
        inbound = (dis_line<radius_soft_thresh) & (min_x<kp[0]<max_x) & (min_y<kp[1]<max_y)
        if inbound:
            sub_graph_id.append(id)

    sub_graph_id = np.asarray(sub_graph_id)
    segment_supports.append(sub_graph_id)

# 3. Show Hungarian - Points & Lines
line_match_table = np.zeros(norm_cross.shape)

for sub_graph_id in segment_supports:
    # print(sub_graph_id)
    if(sub_graph_id.shape[0]>3):
        sub_norm_cross = norm_cross[sub_graph_id,:]
        # his_src_sub = his_src[sub_graph_id]
        # Gs = ot.sinkhorn(his_src_sub, his_tar, sub_norm_cross, reg=1e-1, numItermax=20)
        # match_src, match_id_target = linear_sum_assignment(cost_matrix = 1 - Gs)
        match_src, match_id_target = linear_sum_assignment(cost_matrix=sub_norm_cross)
        match_id = tuple([sub_graph_id[match_src],match_id_target])
        line_match_table[match_id] += 0.1

Gs = ot.sinkhorn(his_src, his_tar, norm_cross, reg=1e-1, numItermax=20)
lb_match = linear_sum_assignment(1 - Gs - line_match_table)
match_score = norm_cross[lb_match]
matches = np.stack((lb_match[0],lb_match[1], match_score), axis=1)
evalScores(pts0, pts1, matches, kpt0_gt, kpt0_valid, thresh=.9)
debug_view = source['color'].copy()

# 1. Line Segments (DONE)
# line_id = 0
# for s, sc in zip(segments, scores):
#     color = tuple(np.random.random(size=3) * 256)
#     thickness = 1
#     txt = str(line_id)
#     line_id += 1
#
#     s = s.astype(np.int32)
#
#     if (sc > 30) :
#         cv.line(debug_view, (s[0], s[1]), (s[2], s[3]), color=color, thickness=thickness, lineType=cv.LINE_AA)
#         org = (int(s[0]/2 + s[2]/2), int(s[1]/2 + s[3]/2))
#         cv.putText(debug_view, text=txt, org=org, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color, thickness=2)
#         cv.drawMarker(debug_view,org,color,markerType=cv.MARKER_SQUARE,markerSize=7,thickness=2)
#
# for id, pt in enumerate(kp0.astype(np.int32)) :
#     his = his_src[id]
#     thickness = 1
#     if his > his_src_thresh:
#         thickness = 2
#     cv.drawMarker(debug_view, pt, (255,0,0), markerType=cv.MARKER_CROSS, markerSize=7, thickness=thickness)
# showRGB(debug_view)
#
# debug_view_tar = target['color'].copy()
# for id, pt in enumerate(kp1.astype(np.int32)) :
#     his = his_tar[id]
#     thickness = 1
#     if his > his_tar_thresh:
#         thickness = 2
#     cv.drawMarker(debug_view_tar, pt, (255,0,0), markerType=cv.MARKER_CROSS, markerSize=7, thickness=thickness)
#
# showRGB(debug_view_tar)