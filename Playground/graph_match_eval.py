import sys
sys.path.append('../')
import numpy as np
from skimage import io
import cv2 as cv
import pyelsed

from config import camera, RGBDImage, PointCloud, TriangleMesh, Image, Vector3dVector, draw_geometries
from config import gtsam, Point2, Point3, Rot3, Pose3, Cal3_S2, Values, PinholeCameraCal3_S2

from Semantics.image_proc_2D import getImgSobel,getSobelMask,showRGB, showDepth, showNorm, \
    getSuperPoints, getSuperPoints_v2
from Semantics.image_proc_3D import cal_Point_Project_TartanCam

from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

from SlamUtils.transformation import pos_quats2SEs, pos_quats2SE_matrices, pose2motion, SEs2ses, line2mat, tartan2kitti
from SlamUtils.visualization import getVisualizationBB, getKeyframe
from SlamUtils.Loader.TartanAir import getRootDir, getDataSequences, getDataLists, tartan_camExtr


rootDIR = getRootDir()
path = getDataSequences(root=rootDIR, scenario='office', level='Easy', seq_num=4)
# path = getDataSequences(root=rootDIR, scenario='seasidetown', level='Easy', seq_num=0)
# path = getDataSequences(root=rootDIR, scenario='neighborhood', level='Easy', seq_num=0)

files_rgb_left, files_rgb_right, files_depth_left, poselist = getDataLists(dir=path, skip=1)
poses_mat44 = pos_quats2SE_matrices(poselist)

config_cam = {'width':640, 'height':480, 'fx':320, 'fy':320, 'cx':320, 'cy':240}
camIntr = camera.PinholeCameraIntrinsic(**config_cam) #open3d if need
camExtr = tartan_camExtr
K = Cal3_S2(320, 320, 0.0, 320, 240)

def getFrameInfo(id):
    frame = {
        'color': io.imread(files_rgb_left[id]),
        'depth': np.load(files_depth_left[id]),
        'transform': poses_mat44[id],
        'intr': camIntr,
        'extr': camExtr
    }
    return frame

def getPoint3D(frame):
    rgbd0 = RGBDImage.create_from_color_and_depth(
        color=Image(frame['color']),
        depth=Image(frame['depth']),
        depth_scale=1.0, depth_trunc=np.inf,
        convert_rgb_to_intensity=False)
    cloud0 = PointCloud.create_from_rgbd_image(image=rgbd0, intrinsic=camIntr, extrinsic=tartan_camExtr)
    cloud0.transform(frame['transform'])
    return np.asarray(cloud0.points)

def getNormMatrices(source_superpoint, target_superpoint):
    desc0 = source_superpoint['desc']
    desc1 = target_superpoint['desc']

    norm_self = np.sqrt(2 - 2 * np.clip(np.dot(desc0.T, desc0), -1, 1))
    norm_cross = np.sqrt(2 - 2 * np.clip(np.dot(desc0.T, desc1), -1, 1))

    return norm_self, norm_cross

def getMatchPrecision(source_id,target_id):
    source = getFrameInfo(source_id)
    target = getFrameInfo(target_id)

    # source_cam = PinholeCameraCal3_S2(Pose3(source['transform']), K)
    target_cam = PinholeCameraCal3_S2(Pose3(target['transform']), K)

    src_gray = cv.cvtColor(source['color'], cv.COLOR_RGB2GRAY)
    src_p3d = getPoint3D(source)

    tar_gray = cv.cvtColor(target['color'], cv.COLOR_RGB2GRAY)

    pts0 = getSuperPoints_v2(src_gray)
    pts1 = getSuperPoints_v2(tar_gray)

    kpt0_gt = []
    kpt0_valid = []

    for kp in pts0['pts']:
        x, y = kp[0], kp[1]
        p0w = Point3(src_p3d[int(y * 640 + x)])
        p0_target, d_prj = cal_Point_Project_TartanCam(target_cam, p0w)
        kpt0_gt.append(p0_target)
        if (0 < p0_target[0] < 640) & (0 < p0_target[1] < 480):
            d_gt = target['depth'][p0_target[1], p0_target[0]]
            dif = abs(d_prj / d_gt)
            if (0.66 < dif < 1.5):
                kpt0_valid.append(True)
            else:
                # print("WARNING DIFF: ", dif, d_gt)
                kpt0_valid.append(False)
        else:
            kpt0_valid.append(False)

    num_matchable = kpt0_valid.count(True)

    norm_self, norm_cross = getNormMatrices(pts0, pts1)
    match_id = linear_sum_assignment(cost_matrix=norm_cross)
    match_score = norm_cross[match_id]
    matches = np.stack((match_id[0], match_id[1], match_score), axis=1)
    idx_sorted = np.argsort(match_score)
    matches = matches[idx_sorted, :]

    kpt0 = pts0['pts'].astype(int)
    kpt1 = pts1['pts'].astype(int)
    kpt0_gt = np.asarray(kpt0_gt)

    match_state = []
    for id0, id1, mVal in matches:
        # p0 = kpt0[int(id0)]
        p0_gt = kpt0_gt[int(id0)]
        valid = kpt0_valid[int(id0)]
        p1 = kpt1[int(id1)]

        dis = np.linalg.norm(p0_gt - p1)
        if (mVal < 0.7):
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
    if(match_state.count('TRUE')>0):
        precision = match_state.count('TRUE') / (match_state.count('TRUE') + match_state.count('FALSE'))
        recall = match_state.count('TRUE') / (match_state.count('TRUE') + match_state.count('SKIP_BUT_VALID'))

    print('Precision = ', precision, '; Recall', recall)
    return precision, recall

# getMatchPrecision(32,36)

step = 10
src_id = 0
tar_id = src_id+step
last_id = len(files_rgb_left)
total_rs = []

while (src_id+20)<last_id:
    p1,r1 = getMatchPrecision(src_id,src_id + 5)
    p2,r2 = getMatchPrecision(src_id, src_id + 10)
    p4,r4 = getMatchPrecision(src_id, src_id + 20)

    rs = [p1,r1,p2,r2,p4,r4]
    total_rs.append(rs)
    src_id+=step

np.savetxt(fname = 'office_004_hg.txt', X=np.asarray(total_rs), header='p1 r1 p2 r2 p4 r4 #steps=[5_10_20]')
# np.savetxt(fname = 'seasidetown_000_hg.txt', X=np.asarray(total_rs), header='p1 r1 p2 r2 p4 r4 #steps=[5_10_20]')
# np.savetxt(fname = 'neighborhood_000_hg.txt', X=np.asarray(total_rs), header='p1 r1 p2 r2 p4 r4 #steps=[5_10_20]')

rs_np = np.asarray(total_rs)
p1 = rs_np[:,0]
r1 = rs_np[:,1]
p2 = rs_np[:,2]
r2 = rs_np[:,3]
p4 = rs_np[:,4]
r4 = rs_np[:,5]

print('Match 05-step: ', p1.mean(),r1.mean())
print('Match 10-step: ', p2.mean(),r2.mean())
print('Match 20-step: ', p4[p4!=0].mean(),r4[r4!=0].mean())

# Remove ZEROS
p4_rf = p4[p4!=0]
r4_rf = r4[r4!=0]

# source = getFrameInfo(350)
# target = getFrameInfo(370)
#
# source_cam = PinholeCameraCal3_S2(Pose3(source['transform']),K)
# target_cam = PinholeCameraCal3_S2(Pose3(target['transform']),K)
#
# src_gray = cv.cvtColor(source['color'], cv.COLOR_RGB2GRAY)
# src_p3d = getPoint3D(source)
#
# tar_gray = cv.cvtColor(target['color'], cv.COLOR_RGB2GRAY)
#
# pts0 = getSuperPoints_v2(src_gray)
# pts1 = getSuperPoints_v2(tar_gray)
#
# num_detected = pts0['pts'].shape[1]
# kpt0_gt = []
# kpt0_valid = []
#
# for kp in pts0['pts']:
#     x,y = kp[0],kp[1]
#     p0w = Point3(src_p3d[int(y * 640 + x)])
#     p0_target,_ = cal_Point_Project_TartanCam(target_cam, p0w)
#     kpt0_gt.append(p0_target)
#     if (0 < p0_target[0] < 640) & (0 <p0_target[1]<480):
#         kpt0_valid.append(True)
#     else:
#         kpt0_valid.append(False)
#
# num_matchable = kpt0_valid.count(True)
#
# norm_self, norm_cross = getNormMatrices(pts0,pts1)
# match_id = linear_sum_assignment(cost_matrix=norm_cross)
# match_score = norm_cross[match_id]
# matches = np.stack((match_id[0],match_id[1], match_score), axis=1)
# idx_sorted = np.argsort(match_score)
# matches = matches[idx_sorted,:]
#
# kpt0 = pts0['pts'].astype(int)
# kpt1 = pts1['pts'].astype(int)
# kpt0_gt = np.asarray(kpt0_gt)
#
# match_state = []
# for id0, id1, mVal in matches:
#     p0 = kpt0[int(id0)]
#     p0_gt = kpt0_gt[int(id0)]
#     valid = kpt0_valid[int(id0)]
#     p1 = kpt1[int(id1)]
#
#     dis = np.linalg.norm(p0_gt - p1)
#     if (mVal<0.7):
#         if(dis < 8):
#             match_state.append('TRUE')
#         else:
#             match_state.append('FALSE')
#     else:
#         if valid:
#             match_state.append('SKIP_BUT_VALID')
#         else:
#             match_state.append('SKIP')
#
# precision, recall, f1 = 0, 0, 0
# if (match_state.count('TRUE') > 0):
#     precision = match_state.count('TRUE') / (match_state.count('TRUE') + match_state.count('FALSE'))
#     recall = match_state.count('TRUE') / (match_state.count('TRUE') + match_state.count('SKIP_BUT_VALID'))
#     f1 = 2 * precision * recall / (precision + recall)
# print('Precision = ', precision, '; Recall = ', recall, '; F1 = ', f1)