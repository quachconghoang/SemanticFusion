import sys
sys.path.append('../')

import numpy as np
import cv2 as cv
from skimage import data, io, filters
from matplotlib import pyplot as plt


from SlamUtils.transformation import pos_quats2SEs, pos_quats2SE_matrices, pose2motion, SEs2ses, line2mat, tartan2kitti
from SlamUtils.visualization import getVisualizationBB, getKeyframe
# from SlamUtils.utils import dataset_intrinsics
from SlamUtils.Loader.TartanAir import getRootDir, getDataSequences, getDataLists, tartan_camExtr

import open3d as o3d
from open3d import camera
from open3d.cuda.pybind.geometry import RGBDImage,PointCloud, TriangleMesh, Image
from open3d.cuda.pybind.utility import Vector3dVector
from open3d.cuda.pybind.visualization import draw_geometries

import gtsam
from gtsam import Point2, Point3, Rot3, Pose3, Cal3_S2, Values
from gtsam import PinholeCameraCal3_S2

rootDIR = getRootDir()
path = getDataSequences(root=rootDIR, scenario='office', level='Easy', seq_num=4)
files_rgb_left, files_rgb_right, files_depth_left, poselist = getDataLists(dir=path, skip=5)

config_cam = {'width':640, 'height':480, 'fx':320, 'fy':320, 'cx':320, 'cy':240}
camIntr = camera.PinholeCameraIntrinsic(**config_cam) #open3d if need

def cal_Point_Project_TartanCam(cam: PinholeCameraCal3_S2, pw: Point3):
    q = cam.pose().transformTo(pw)[[1,2,0]] #
    pn = cam.Project(q)
    pi = cam.calibration().uncalibrate(pn)
    # d = 1 / q[2]
    return np.rint(pi).astype(np.int32)

# poses_mat34 = pos_quats2SEs(poselist)  # [R|t - array 12]
# poses_mat34_kitty, poses_mat44_kitty = tartan2kitti(poselist)
poses_mat44 = pos_quats2SE_matrices(poselist)
# motions_mat = pose2motion(poses_mat34)  # [R|t]
# motions_quat = SEs2ses(motions_mat).astype(np.float32)  # x-y-z qx-qy-qz-qw

id = 26
### ----- DATA ----- ###
frame0 = {
    'color':io.imread(files_rgb_left[id]),
    'depth':np.load(files_depth_left[id]),
    'transform':poses_mat44[id]
}
id = 32
frame1 = {
    'color':io.imread(files_rgb_left[id]),
    'depth':np.load(files_depth_left[id]),
    'transform':poses_mat44[id]
}

### ------ VIZ ------ ###
config_viz ={'width':1024, 'height':768,
             'zoom':0.5, 'front':[0,-1,-.25],
             'lookat':[0,0,0], 'up':[0,0,-1]}
viz_base = [getVisualizationBB(),TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])]
viz_cam = [getKeyframe(transform=frame0['transform'],color=[1,0,0]),
           getKeyframe(transform=frame1['transform'],color=[1,0,0])]

viz = []
viz.extend(viz_base)
viz.extend(viz_cam)
# draw_geometries(viz, **config_viz)


# for pose in poses_mat44:
#     viz_cam.append(getKeyframe(transform=pose,color=[1,0,0]))

# ----- GTSAM -----
# plt.imshow(frame0['color']);plt.show()
# plt.imshow(frame1['color']);plt.show()

K = Cal3_S2(320, 320, 0.0, 320, 240)
pose0 = Pose3(frame0['transform'])
cam0 = PinholeCameraCal3_S2(pose0,K)
pose1 = Pose3(frame1['transform'])
cam1 = PinholeCameraCal3_S2(pose1,K)

rgbd0 = RGBDImage.create_from_color_and_depth(
                color=Image(frame0['color']),
                depth=Image(frame0['depth']),
                depth_scale=1.0, depth_trunc=np.inf,
                convert_rgb_to_intensity=False)
cloud0 = PointCloud.create_from_rgbd_image(image=rgbd0, intrinsic=camIntr, extrinsic=tartan_camExtr)
cloud0.transform(frame0['transform'])
point3D = np.asarray(cloud0.points)

# draw_geometries(viz, **config_viz)
def checkInBox(pt, tl=[0,0], br=[640,480]):
    if((pt[0] >= tl[0]) &(pt[0]<br[0]) &(pt[1]>=tl[1]) &(pt[1]<br[1])):
        return True
    else:
        return False

def cvShowRGB(img, name = 'cv-show'):
    out = cv.cvtColor(img,cv.COLOR_RGB2BGR)
    cv.imshow(name,out);cv.waitKey();cv.destroyAllWindows()

### --- Match Points ---
### Check neighbor depth
img = frame0['color'].copy()
img_target = frame1['color'].copy()
img_bg = np.concatenate((img,img_target),axis=1)
img_overlay = np.zeros(img_bg.shape,dtype=np.uint8)

gray = cv.cvtColor(frame0['color'], cv.COLOR_RGB2GRAY)
corners = cv.goodFeaturesToTrack(gray,25,0.01,10)
corners = np.int0(corners)
for i in corners:
    x,y = i.ravel()
    color = tuple(np.random.random(size=3) * 256)
    colorRED = [255,0,0]
    p0w = Point3(point3D[y * 640 + x])
    p0_target = cal_Point_Project_TartanCam(cam1, p0w)

    cv.circle(img_overlay, (x, y), radius=5, color=color, thickness=2)

    if(checkInBox(p0_target,[0,0],[640,480])):
        cv.circle(img_overlay, p0_target+(640,0), radius=5, color=color, thickness=2)
        cv.line(img_overlay, (x, y), p0_target+(640,0), color=colorRED, thickness=1, lineType=cv.LINE_AA)

    # img_target = cv.drawMarker(img_target, p0_target, color, cv.MARKER_CROSS, 5)

img_preview = cv.addWeighted(img_bg, 1, img_overlay, 0.8, 0)
cvShowRGB(img_preview)

# plt.imshow(img_preview),plt.show()
# io.imshow(frame0['depth']),io.show()
# plt.imshow(img_target),plt.show()
# cv.imshow('depth',frame0['depth']/10);cv.waitKey();cv.destroyAllWindows()

### --- Match Lines ---
import pyelsed
inp = cv.cvtColor(frame0['color'], cv.COLOR_RGB2GRAY)
inp = cv.equalizeHist(inp)

segments, scores = pyelsed.detect(inp, gradientThreshold=30, minLineLen=15, lineFitErrThreshold=0.5)
id = np.argsort(scores)[::-1]
segments = segments[id]
scores = scores[id]
img_prev_line =  frame0['color'].copy()
img_match =  frame1['color'].copy()

for s, sc in zip(segments.astype(np.int32), scores):
    color = tuple(np.random.random(size=3) * 256)
    thickness = 2
    if sc > 50:
        cv.line(img_prev_line, (s[0], s[1]), (s[2], s[3]), color, thickness, cv.LINE_AA)
        # print(s) #PLEASE CHECK!
        p0w = Point3(point3D[s[1]*640+s[0]])
        p1w = Point3(point3D[s[3]*640+s[2]])
        p0 = cal_Point_Project_TartanCam(cam1,p0w)
        p1 = cal_Point_Project_TartanCam(cam1,p1w)
        cv.line(img_match, p0, p1, color, thickness, cv.LINE_AA)
        # cv.drawMarker(img_prev_line, (s[0], s[1]), [255, 0, 0], cv.MARKER_CROSS, 5)
        # cv.drawMarker(img_prev_line, (s[2], s[3]), [255, 0, 0], cv.MARKER_CROSS, 5)

plt.imshow(img_prev_line);plt.show()
plt.imshow(img_match);plt.show()

# TODO:
#  - Fix bugs with wrong depth pixel - wrong depth => Line-encoding methods
#  - Solution Gen 3D lines -> Line models -> RANSAC -> Real 3D Lines points
#  - Bug Test: Steps = 5, id_0 = 28; id_1=32