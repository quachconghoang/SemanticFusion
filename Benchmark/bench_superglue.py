import sys
sys.path.append('../')

import numpy as np
import cv2 as cv
from skimage import io
import json

from Semantics.SuperGlue.models.utils import (make_matching_plot_fast, frame2tensor)
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

# path = db['office']['Easy'][0]
# files_rgb_left, files_rgb_right, files_depth_left, poselist = getDataLists(dir=path, skip=5)
# poses_mat44 = pos_quats2SE_matrices(poses_quad)

def getFrameInfo(id):
    frame = {   'color': io.imread(files_rgb_left[id]),
                # 'color_right': io.imread(files_rgb_right[id]),
                'depth': np.load(files_depth_left[id]),
                'transform': poses_mat44[id],
                'intr': camIntr,
                'extr': camExtr     }
    return frame

def getMatchPrecision(source_id,target_id):
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
    frame_tensor = frame2tensor(tar_gray, dnn_device)
    pred = matching({**last_data, 'image1': frame_tensor})
    kpts0 = last_data['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()
    confidence = pred['matching_scores0'][0].cpu().numpy()

    kpt0_gt = []
    kpt0_valid = []

    for kp in kpts0:
        x, y = kp
        p0w = Point3(src_p3d[int(y * 640 + x)])
        p0_target, d_prj = cal_Point_Project_TartanCam(target_cam, p0w)
        kpt0_gt.append(p0_target)
        if (0 < p0_target[0] < 640) & (0 < p0_target[1] < 480):
            d_gt = target['depth'][p0_target[1],p0_target[0]]
            dif = abs(d_prj/d_gt)
            if (0.8 < dif < 1.25):
                kpt0_valid.append(True)
            else:
                # print("WARNING DIFF: ", dif, d_gt)
                kpt0_valid.append(False)
        else:
            kpt0_valid.append(False)


    match_state = []
    for id0, id1 in enumerate(matches):
        p0_gt = kpt0_gt[int(id0)]
        valid = kpt0_valid[int(id0)]
        p1 = kpts1[int(id1)]
        dis = np.linalg.norm(p0_gt - p1)
        if (id1 > -1):
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

    # print('Precision = ', precision, '; Recall = ', recall,  '; F1 = ', f1)
    return precision, recall, f1

# paths = []
# paths.extend(db['office']['Easy'])
# paths.extend(db['office']['Hard'])

for path in db['office']['Hard']:

    files_rgb_left, files_rgb_right, files_depth_left, poses_quad = getDataLists(dir=path, skip=5)
    poses_mat44 = pos_quats2SE_matrices(poses_quad)

    step = 2
    src_id = 0
    last_id = len(files_rgb_left)

    total_rs = []

    while (src_id + 4) < last_id:
        p4, r4, f4 = getMatchPrecision(src_id, src_id + 4)
        rs = [p4, r4, f4]
        total_rs.append(rs)
        src_id += step

    rs_np = np.asarray(total_rs)
    p4 = rs_np[:, 0]
    r4 = rs_np[:, 1]
    f4 = rs_np[:, 2]

    print(path)
    print('Match 20-step:',  '%6f' % p4[p4 != 0].mean(), '%6f' % r4[r4 != 0].mean(), '%6f' % f4[f4 != 0].mean())

