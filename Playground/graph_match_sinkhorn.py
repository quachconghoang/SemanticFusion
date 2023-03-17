import sys

sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from skimage import io
import cv2 as cv
import ot
# import pyelsed

from scipy.optimize import linear_sum_assignment

from config import camera, RGBDImage, PointCloud, TriangleMesh, Image, Vector3dVector, draw_geometries
from config import gtsam, Point2, Point3, Rot3, Pose3, Cal3_S2, Values, PinholeCameraCal3_S2

from SlamUtils.Loader.TartanAir import rootDIR, getDataSequences, getDataLists, tartan_camExtr
from SlamUtils.transformation import pos_quats2SEs, pos_quats2SE_matrices, pose2motion, SEs2ses, line2mat, tartan2kitti

from Semantics.image_proc_3D import cal_Point_Project_TartanCam, generate3D
from Semantics.image_proc_2D import getSuperPoints_v2, hungarianMatch,sinkhornMatch,\
    getImgSobel,getSobelMask,showRGB, showDepth, showNorm

path = getDataSequences(root=rootDIR, scenario='office', level='Easy', seq_num=4)
# path = getDataSequences(root=rootDIR, scenario = 'seasidetown', level='Easy', seq_num=0)
# path = getDataSequences(root=rootDIR, scenario = 'neighborhood', level='Easy', seq_num=0)
files_rgb_left, files_rgb_right, files_depth_left, poselist = getDataLists(dir=path, skip=5)
poses_mat44 = pos_quats2SE_matrices(poselist)

config_cam = {'width':640, 'height':480, 'fx':320, 'fy':320, 'cx':320, 'cy':240}
camIntr = camera.PinholeCameraIntrinsic(**config_cam) #open3d if need
camExtr = tartan_camExtr
K = Cal3_S2(320, 320, 0.0, 320, 240)

def getFrameInfo(id):
    frame = {   'color': io.imread(files_rgb_left[id]),
                'depth': np.load(files_depth_left[id]),
                'transform': poses_mat44[id],
                'intr': camIntr,
                'extr': camExtr     }
    return frame

# def hungarianMatch(norm_cross):
#     match_id = linear_sum_assignment(cost_matrix = norm_cross)
#     match_score = norm_cross[match_id]
#     matches = np.stack((match_id[0], match_id[1], match_score), axis=1)
#     idx_sorted = np.argsort(match_score)
#     matches = matches[idx_sorted, :]
#     return matches
#
# def sinkhornMatch(norm_cross, norm_src, norm_tar, lambd=1e-1, iter=20):
#     a, b = [], []
#     for pt_his in norm_src:
#         a.append(pt_his.mean() - 1)
#     for pt_his in norm_tar:
#         b.append(pt_his.mean() - 1)
#
#     Gs = ot.sinkhorn(a, b, norm_cross, reg=lambd, numItermax=iter)
#     matches = hungarianMatch(1 - Gs)
#     return matches

def getMatchPrecision(source_id,target_id):
    source = getFrameInfo(source_id)
    target = getFrameInfo(target_id)

    generate3D(source)
    src_p3d = source['point3D']
    target_cam = PinholeCameraCal3_S2(Pose3(target['transform']), K)

    src_gray = cv.cvtColor(source['color'], cv.COLOR_RGB2GRAY)
    tar_gray = cv.cvtColor(target['color'], cv.COLOR_RGB2GRAY)

    pts0 = getSuperPoints_v2(src_gray)
    pts1 = getSuperPoints_v2(tar_gray)

    desc0 = pts0['desc']
    desc1 = pts1['desc']

    norm_self_src = np.sqrt(2 - 2 * np.clip(np.dot(desc0.T, desc0), -1, 1))
    norm_self_tar = np.sqrt(2 - 2 * np.clip(np.dot(desc1.T, desc1), -1, 1))
    norm_cross = np.sqrt(2 - 2 * np.clip(np.dot(desc0.T, desc1), -1, 1))

    num_detected = pts0['pts'].shape[0]
    kpt0_gt = []
    kpt0_valid = []

    for kp in pts0['pts']:
        x, y = kp[0], kp[1]
        p0w = Point3(src_p3d[int(y * 640 + x)])
        p0_target, _ = cal_Point_Project_TartanCam(target_cam, p0w)
        kpt0_gt.append(p0_target)
        if (0 < p0_target[0] < 640) & (0 < p0_target[1] < 480):
            kpt0_valid.append(True)
        else:
            kpt0_valid.append(False)

    kpt0 = pts0['pts'].astype(int)
    kpt1 = pts1['pts'].astype(int)
    kpt0_gt = np.asarray(kpt0_gt)

    matches = sinkhornMatch(norm_cross, norm_self_src, norm_self_tar)

    match_state = []
    for id0, id1, mVal in matches:
        # p0 = kpt0[int(id0)]
        p0_gt = kpt0_gt[int(id0)]
        valid = kpt0_valid[int(id0)]
        p1 = kpt1[int(id1)]

        dis = np.linalg.norm(p0_gt - p1)
        if (mVal < 0.75):
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
    if (match_state.count('TRUE') > 0):
        precision = match_state.count('TRUE') / (match_state.count('TRUE') + match_state.count('FALSE'))
        recall = match_state.count('TRUE') / (match_state.count('TRUE') + match_state.count('SKIP_BUT_VALID'))

    print('Precision = ', precision, '; Recall', recall)
    return precision, recall

step = 2
src_id = 0
tar_id = src_id+step
last_id = len(files_rgb_left)
total_rs = []

while (src_id+4)<last_id:
    p1,r1 = getMatchPrecision(src_id,src_id + 1)
    p2,r2 = getMatchPrecision(src_id, src_id + 2)
    p4,r4 = getMatchPrecision(src_id, src_id + 4)

    rs = [p1,r1,p2,r2,p4,r4]
    total_rs.append(rs)
    src_id+=step

np.savetxt(fname = 'office_004_sh.txt', X=np.asarray(total_rs), header='p1 r1 p2 r2 p4 r4 #steps=[5_10_20]')
# np.savetxt(fname = 'neighborhood_000_sh.txt', X=np.asarray(total_rs), header='p1 r1 p2 r2 p4 r4 #steps=[5_10_20]')
rs_np = np.asarray(total_rs)
p1 = rs_np[:,0]
r1 = rs_np[:,1]
p2 = rs_np[:,2]
r2 = rs_np[:,3]
p4 = rs_np[:,4]
r4 = rs_np[:,5]

print('Match 05-step: ', p1.mean(),r1.mean())
print('Match 10-step: ', p2[p2!=0].mean(),r2[r2!=0].mean())
print('Match 20-step: ', p4[p4!=0].mean(),r4[r4!=0].mean())

source = getFrameInfo(32)
target = getFrameInfo(36)

generate3D(source)
src_p3d = source['point3D']
target_cam = PinholeCameraCal3_S2(Pose3(target['transform']), K)

src_gray = cv.cvtColor(source['color'], cv.COLOR_RGB2GRAY)
tar_gray = cv.cvtColor(target['color'], cv.COLOR_RGB2GRAY)

pts0 = getSuperPoints_v2(src_gray)
pts1 = getSuperPoints_v2(tar_gray)

sp0 = pts0['pts']; desc0 = pts0['desc']
sp1 = pts1['pts']; desc1 = pts1['desc']

norm_self_src = np.sqrt(2 - 2 * np.clip(np.dot(desc0.T, desc0), -1, 1))
norm_self_tar = np.sqrt(2 - 2 * np.clip(np.dot(desc1.T, desc1), -1, 1))
norm_cross = np.sqrt(2 - 2 * np.clip(np.dot(desc0.T, desc1), -1, 1))
# showNorm(norm_self_src)
# showNorm(norm_self_tar)
# showNorm(norm_cross)

# - 3[-.-]5
num_detected = pts0['pts'].shape[0]
kpt0_gt = []
kpt0_valid = []

for kp in pts0['pts']:
    x,y = kp[0],kp[1]
    p0w = Point3(src_p3d[int(y * 640 + x)])
    p0_target,_ = cal_Point_Project_TartanCam(target_cam, p0w)
    kpt0_gt.append(p0_target)
    if (0 < p0_target[0] < 640) & (0 <p0_target[1]<480):
        kpt0_valid.append(True)
    else:
        kpt0_valid.append(False)

kpt0 = pts0['pts'].astype(int)
kpt1 = pts1['pts'].astype(int)
kpt0_gt = np.asarray(kpt0_gt)

src0 = pts0['scores']
src1 = pts1['scores']

matches = sinkhornMatch(norm_cross,norm_self_src,norm_self_tar)

match_state = []
for id0, id1, mVal in matches:
    # p0 = kpt0[int(id0)]
    p0_gt = kpt0_gt[int(id0)]
    valid = kpt0_valid[int(id0)]
    p1 = kpt1[int(id1)]

    dis = np.linalg.norm(p0_gt - p1)
    if (mVal < 0.75):
        if (dis < 8):
            match_state.append('TRUE')
        else:
            match_state.append('FALSE')
    else:
        if valid:
            match_state.append('SKIP_BUT_VALID')
        else:
            match_state.append('SKIP')

precision = match_state.count('TRUE')/(match_state.count('TRUE')+match_state.count('FALSE'))
recall = match_state.count('TRUE')/(match_state.count('TRUE')+match_state.count('SKIP_BUT_VALID'))
print('Precision = ', precision, '; Recall', recall)

# 0.53 - 0.702