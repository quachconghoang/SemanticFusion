import sys
sys.path.append('../')

from SlamUtils.transformation import pos_quats2SEs, pos_quats2SE_matrices, pose2motion, SEs2ses, line2mat, tartan2kitti
from SlamUtils.visualization import getVisualizationBB, getKeyframe
from SlamUtils.utils import dataset_intrinsics
from SlamUtils.Loader.TartanAir import getRootDir, getDataSequences, getDataLists
from SlamUtils.Loader.TartanAir import rootDIR, getDataSequences, getDataLists, tartan_camExtr


from pathlib import Path
import numpy as np
import matplotlib.cm as cm
import torch
import cv2 as cv
from skimage import data, io, filters

# from Semantics.SuperGlue.models.matching import Matching
from Semantics.SuperGlue.models.utils import (make_matching_plot_fast, frame2tensor)
from Semantics.image_proc_2D import matching, dnn_device
from Semantics.image_proc_3D import cal_Point_Project_TartanCam

from config import camera, RGBDImage, PointCloud, TriangleMesh, Image, Vector3dVector, draw_geometries
from config import gtsam, Point2, Point3, Rot3, Pose3, Cal3_S2, Values, PinholeCameraCal3_S2


path = getDataSequences(root=rootDIR, scenario='office', level='Easy', seq_num=4)
# path = getDataSequences(root=rootDIR, scenario='seasidetown', level='Easy', seq_num=0)
# path = getDataSequences(root=rootDIR, scenario='neighborhood', level='Easy', seq_num=0)
files_rgb_left, files_rgb_right, files_depth_left, poselist = getDataLists(dir=path, skip=5)
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
            if (0.66 < dif < 1.5):
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

        # print(id0,id1, p1,p0_gt,valid)
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

    print('Precision = ', precision, '; Recall = ', recall,  '; F1 = ', f1)
    return precision, recall, f1

step = 2
src_id = 0
tar_id = src_id+step
last_id = len(files_rgb_left)
total_rs = []

while (src_id+4)<last_id:
    # p1,r1 = getMatchPrecision(src_id,src_id + 1)
    # p2,r2 = getMatchPrecision(src_id, src_id + 2)
    p4,r4,f4 = getMatchPrecision(src_id, src_id + 4)

    rs = [p4,r4,f4]
    total_rs.append(rs)
    src_id+=step

# np.savetxt(fname = 'office_004_sg.txt', X=np.asarray(total_rs), header='p1 r1 p2 r2 p4 r4 #steps=[5_10_20]=superglue')
# np.savetxt(fname = 'seasidetown_000_sg.txt', X=np.asarray(total_rs), header='p1 r1 p2 r2 p4 r4 #steps=[5_10_20]=superglue')
# np.savetxt(fname = 'neighborhood_000_sg.txt', X=np.asarray(total_rs), header='p1 r1 p2 r2 p4 r4 #steps=[5_10_20]=superglue')

rs_np = np.asarray(total_rs)
p4 = rs_np[:,0]
r4 = rs_np[:,1]
f4 = rs_np[:,2]

# print('Match 05-step: ', p1.mean(),r1.mean())
# print('Match 10-step: ', p2.mean(),r2.mean())
print('Match 20-step: ', p4[p4!=0].mean(),r4[r4!=0].mean(), f4[f4!=0].mean())

source = getFrameInfo(32)
target = getFrameInfo(36)

source_cam = PinholeCameraCal3_S2(Pose3(source['transform']),K)
target_cam = PinholeCameraCal3_S2(Pose3(target['transform']),K)

src_gray = cv.cvtColor(source['color'], cv.COLOR_RGB2GRAY)
src_p3d = getPoint3D(source)
tar_gray = cv.cvtColor(target['color'], cv.COLOR_RGB2GRAY)

# FRAME 0
frame_tensor = frame2tensor(src_gray, dnn_device)

keys = ['keypoints', 'scores', 'descriptors']
last_data = matching.superpoint({'image': frame_tensor})
last_data = {k+'0': last_data[k] for k in keys}
last_data['image0'] = frame_tensor
last_frame = src_gray

# FRAME 1
frame_tensor = frame2tensor(tar_gray, dnn_device)
pred = matching({**last_data, 'image1': frame_tensor})
kpts0 = last_data['keypoints0'][0].cpu().numpy()
kpts1 = pred['keypoints1'][0].cpu().numpy()
matches = pred['matches0'][0].cpu().numpy()
confidence = pred['matching_scores0'][0].cpu().numpy()

num_detected = kpts0.shape[1]
kpt0_gt = []
kpt0_valid = []

for kp in kpts0:
    x,y = kp
    p0w = Point3(src_p3d[int(y * 640 + x)])
    p0_target,_ = cal_Point_Project_TartanCam(target_cam, p0w)
    kpt0_gt.append(p0_target)
    if (0 < p0_target[0] < 640) & (0 <p0_target[1]<480):
        kpt0_valid.append(True)
    else:
        kpt0_valid.append(False)

match_state = []
for id0,id1 in enumerate(matches):
    p0_gt = kpt0_gt[int(id0)]
    valid = kpt0_valid[int(id0)]
    p1 = kpts1[int(id1)]
    dis = np.linalg.norm(p0_gt - p1)

    # print(id0,id1, p1,p0_gt,valid)
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
if(match_state.count('TRUE')>0):
    precision = match_state.count('TRUE') / (match_state.count('TRUE') + match_state.count('FALSE'))
    recall = match_state.count('TRUE') / (match_state.count('TRUE') + match_state.count('SKIP_BUT_VALID'))

print('Precision = ', precision, '; Recall = ', recall,  '; F1 = ', 2*precision*recall/(precision+recall))