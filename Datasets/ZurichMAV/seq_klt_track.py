import glob
import numpy as np
import open3d as o3d
import cv2 as cv

from Semantics.image_proc_2D import matching, dnn_device, getSuperPoints_v2, getAnchorPoints

from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 400

import sys
sys.path.append('../')

img_path = '/home/hoangqc/Datasets/AGZ/MAV_Images/'
mav_gt_file = '/home/hoangqc/Datasets/AGZ/Log Files/GroundTruthAGL.csv'
img_distCoeff = np.load('/home/hoangqc/Datasets/AGZ/calibration_data/distCoeff.npy')[0]
img_intrinsic = np.load('/home/hoangqc/Datasets/AGZ/calibration_data/intrinsic_matrix.npy')
def_intrinsics = np.zeros((3, 4))
def_intrinsics[:3, :3] = img_intrinsic

def get_image_files(folder, ext='jpg'):
    files = glob.glob(folder + '*.' +ext)
    files.sort()
    return files

def draw_keypoints(img, kp, color=(0, 255, 0), marker = cv.MARKER_CROSS):
    for i in range(len(kp)):
        cv.drawMarker(img, (int(kp[i, 0]), int(kp[i, 1])), color, marker, 10, 2)
    return img

# Get data from CSV file:
def get_data_from_csv(filename):
    data = np.genfromtxt(filename, delimiter=',')
    return data

imgs = get_image_files(img_path)

key_id = 2000
start_id = key_id*30
distance = 120


src = cv.imread(imgs[start_id], cv.IMREAD_UNCHANGED)
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
pts0 = getSuperPoints_v2(src_gray)
kp0 = pts0['pts']
desc0 = pts0['desc']

lk_params = dict(winSize=(33, 33),
                 maxLevel=3,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))

img_last = src_gray.copy()
kp_last = kp0.copy()
st = np.zeros((kp_last.shape[0], 1), dtype=np.uint8)
kp_id = np.arange(start=0, stop=kp_last.shape[0]  ,dtype=int)


for i in range(distance):
    tar = cv.imread(imgs[start_id + 1 + i], cv.IMREAD_UNCHANGED)
    tar_gray = cv.cvtColor(tar, cv.COLOR_BGR2GRAY)
    p1, st, err = cv.calcOpticalFlowPyrLK(img_last, tar_gray, kp_last, None, **lk_params)
    st = st.squeeze()
    # Update new keypoints
    kp_id = kp_id[np.where(st == 1)]
    kp_last = p1[st == 1]

    img_last = tar_gray.copy()

kp_first = kp0[kp_id]
kp_first_uv = cv.undistortPoints(kp_first, cameraMatrix=img_intrinsic, distCoeffs=img_distCoeff, P=def_intrinsics).squeeze()
kp_last_uv = cv.undistortPoints(kp_last, cameraMatrix=img_intrinsic, distCoeffs=img_distCoeff, P=def_intrinsics).squeeze()
M, mask = cv.findHomography(kp_first_uv, kp_last_uv, cv.RANSAC, 64.0, confidence=0.9, maxIters=1000)

kp_first = kp_first[mask.squeeze() == 1]
kp_last = kp_last[mask.squeeze() == 1]

vis_src = draw_keypoints(src, kp0, color=(0, 255, 0))
vis_tar = draw_keypoints(tar, kp_last, color=(255, 255, 0), marker=cv.MARKER_SQUARE)

# Draw matches keypoints
def drawMatchKeypoints(src_img, tar_img, src_kp, tar_kp):
    vis = np.concatenate((src_img, tar_img), axis=1)
    for index in range(100):
        i = np.random.randint(0, src_kp.shape[0])
        color = list(np.random.random(size=3) * 256)
        cv.drawMarker(vis, (int(src_kp[i, 0]), int(src_kp[i, 1])), (0, 255, 0), cv.MARKER_CROSS, 10, 2)
        cv.drawMarker(vis, (int(tar_kp[i, 0] + src_img.shape[1]), int(tar_kp[i, 1])), (0, 255, 0), cv.MARKER_SQUARE, 10, 2)
        cv.line(vis, (int(src_kp[i, 0]), int(src_kp[i, 1])), (int(tar_kp[i, 0] + src_img.shape[1]), int(tar_kp[i, 1])),color=color, thickness=2)
    return vis

vis = drawMatchKeypoints(src, tar, kp_first, kp_last)
plt.imshow(vis)
plt.show()


cv.imwrite('matches-'+ str(key_id) +'.png', vis)