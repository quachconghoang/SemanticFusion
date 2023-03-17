import sys
sys.path.append('../')
from pathlib import Path
import numpy as np
import matplotlib.cm as cm
import torch
import cv2 as cv
from skimage import data, io, filters

import time

from SlamUtils.transformation import pos_quats2SEs, pos_quats2SE_matrices, pose2motion, SEs2ses, line2mat, tartan2kitti
from SlamUtils.Loader.TartanAir import rootDIR, getDataSequences, getDataLists, tartan_camExtr

# from Semantics.SuperGlue.models.matching import Matching
from Semantics.SuperGlue.models.utils import frame2tensor
from Semantics.image_proc_2D import getSuperPoints_v2, hungarianMatch, sinkhornMatch, \
    getImgSobel,getSobelMask,showRGB, showDepth, showNorm
from Semantics.image_proc_2D import matching, dnn_device
from Semantics.image_proc_3D import generate3D, getPointsProject2D, \
    cal_Point_Project_General, cal_Point_Project_TartanCam, \
    getDisparityTartanAIR, get3DLocalFromDisparity

from config import camera, RGBDImage, PointCloud, TriangleMesh, Image, Vector3dVector, draw_geometries
from config import gtsam, Point2, Point3, Rot3, Pose3, Cal3_S2, Values, PinholeCameraCal3_S2


path = getDataSequences(root=rootDIR, scenario='office', level='Easy', seq_num=4)
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


def getMatchSuperglue(src_id, tar_id):
    source = getFrameInfo(src_id) #(30 * 5)  # 32*5
    target = getFrameInfo(tar_id) #(34 * 5)  # 36*5
    generate3D(source)

    source_cam = PinholeCameraCal3_S2(Pose3(source['transform']),K)
    target_cam = PinholeCameraCal3_S2(Pose3(target['transform']),K)

    src_gray = cv.cvtColor(source['color'], cv.COLOR_RGB2GRAY)
    tar_gray = cv.cvtColor(target['color'], cv.COLOR_RGB2GRAY)

    ### Timing with ...
    frame_tensor_src = frame2tensor(src_gray, dnn_device)
    keys = ['keypoints', 'scores', 'descriptors']
    last_data = matching.superpoint({'image': frame_tensor_src})
    last_data = {k+'0': last_data[k] for k in keys}
    last_data['image0'] = frame_tensor_src
    # last_frame = src_gray

    frame_tensor_tar = frame2tensor(tar_gray, dnn_device)
    current_data = matching.superpoint({'image': frame_tensor_tar})
    current_data = {k+'1': current_data[k] for k in keys}
    current_data['image1'] = frame_tensor_tar

    st = time.time()
    pred = matching({**last_data, **current_data})
    et = time.time()
    matching_time = et - st
    print('Match Execution time:', matching_time, 'seconds')

    # st = time.time()
    # pred = matching({'image0':frame_tensor_src, 'image1': frame_tensor_tar})
    # et = time.time()
    # elapsed_time = et - st
    # print('Full Execution time:', elapsed_time, 'seconds')
    return matching_time

step = 10
src_id = 0
last_id = len(files_rgb_left)
total_rs = []

while (src_id+20)<last_id:
    # p1,r1,f1 = getMatchPrecision(src_id,src_id + 5)
    # p2,r2,f2 = getMatchPrecision(src_id, src_id + 10)
    mt = getMatchSuperglue(src_id, src_id + 20)

    # rs = [p1,r1,f1,p2,r2,f2,p4,r4,f4]
    total_rs.append(mt)
    src_id+=step

print('Final match time: ', np.asarray(total_rs).mean())
# source = getFrameInfo(32*5) # 32*5
# target = getFrameInfo(36*5) # 36*5
# generate3D(source)
#
# source_cam = PinholeCameraCal3_S2(Pose3(source['transform']),K)
# target_cam = PinholeCameraCal3_S2(Pose3(target['transform']),K)
#
# src_gray = cv.cvtColor(source['color'], cv.COLOR_RGB2GRAY)
# tar_gray = cv.cvtColor(target['color'], cv.COLOR_RGB2GRAY)
#
# ### Timing with ...
# frame_tensor_src = frame2tensor(src_gray, dnn_device)
# keys = ['keypoints', 'scores', 'descriptors']
# last_data = matching.superpoint({'image': frame_tensor_src})
# last_data = {k+'0': last_data[k] for k in keys}
# last_data['image0'] = frame_tensor_src
# last_frame = src_gray
#
# frame_tensor_tar = frame2tensor(tar_gray, dnn_device)
# current_data = matching.superpoint({'image': frame_tensor_tar})
# current_data = {k+'1': current_data[k] for k in keys}
# current_data['image1'] = frame_tensor_tar
#
# st = time.time()
# pred = matching({**last_data, **current_data})
# et = time.time()
# elapsed_time = et - st
# print('Match Execution time:', elapsed_time, 'seconds')
#
# st = time.time()
# pred = matching({'image0':frame_tensor_src, 'image1': frame_tensor_tar})
# et = time.time()
# elapsed_time = et - st
# print('Full Execution time:', elapsed_time, 'seconds')

