import sys
sys.path.append('../')

import numpy as np
import cv2 as cv
from skimage import io

from config import gtsam, Point2, Point3, Rot3, Pose3, Cal3_S2, Values, PinholeCameraCal3_S2
from config import camera, RGBDImage, PointCloud,  Image

camExtr = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
camIntr = camera.PinholeCameraIntrinsic(width=640, height=480, fx=320, fy=320, cx=320, cy=240)


def getPoint3D(frame):
    rgbd0 = RGBDImage.create_from_color_and_depth(
        color=Image(frame['color']),
        depth=Image(frame['depth']),
        depth_scale=1.0, depth_trunc=np.inf,
        convert_rgb_to_intensity=False)
    cloud0 = PointCloud.create_from_rgbd_image(image=rgbd0, intrinsic=frame['intr'], extrinsic=frame['extr'])
    cloud0.transform(frame['transform'])
    return np.asarray(cloud0.points)

def evalScores(pts0, pts1, matches, kp0_gt, kp0_valid, thresh=.9):
    kpt1 = pts1['pts'].astype(int)
    match_state = []
    for id0, id1, mVal in matches:
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

# def getFrameInfo(id):
#     frame = {
#         'color': io.imread(files_rgb_left[id]),
#         'depth': np.load(files_depth_left[id]),
#         'transform': poses_mat44[id],
#         'intr': camIntr,
#         'extr': camExtr
#     }
#     return frame