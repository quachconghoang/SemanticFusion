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

source = getFrameInfo(32)
target = getFrameInfo(36)
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


norm_cross = np.sqrt(2 - 2 * np.clip(np.dot(desc0.T, desc1), -1, 1))
match_id = linear_sum_assignment(cost_matrix=(norm_cross))
match_score = norm_cross[match_id]
matches = np.stack((match_id[0], match_id[1], match_score), axis=1)

kpt1 = kpts1.astype(int)
thresh = 0.4
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

print('Precision = ', precision, '; Recall = ', recall,  '; F1 = ', f1)

# concentrate images
gap_mat = np.full((480, 5),fill_value=255, dtype=np.uint8)
vis_img = np.concatenate((src_gray,gap_mat, tar_gray), axis=1)
vis_img = cv.cvtColor(vis_img, cv.COLOR_GRAY2RGB)

# drap matches
count = 0
for id0, id1, mVal in matches:
    p0 = kpts0[int(id0)].astype(int)
    p0_gt = kp0_gt[int(id0)]
    valid = kp0_valid[int(id0)]
    p1 = kpt1[int(id1)].astype(int)
    state = match_state[int(count)]

    if (state == 'FALSE'):
        cv.line(vis_img,p0,(p1[0]+645,p1[1]),(0,0,255),1)
        print(p0,p1)

    if (state == 'TRUE'):
        cv.line(vis_img,p0,(p1[0]+645,p1[1]),(0,255,0),1)
        print(p0,p1)

    count += 1

    # if count > 200:
    #     break
    ...