import sys, os, glob
import numpy as np
import cv2 as cv

import open3d as o3d
from open3d import camera
from open3d.cuda.pybind.geometry import RGBDImage,PointCloud
from open3d.cuda.pybind.utility import Vector3dVector

import pyzed.sl as sl

from matplotlib import pyplot as plt
from skimage import data, io, filters

from Datasets.ZED.util import quats_to_matrices, getPoint3D
from gtsam import Point2, Point3, Rot3, Pose3, Cal3_S2, Values, PinholeCameraCal3_S2
from Semantics.SuperGlue.models.utils import frame2tensor
from Semantics.image_proc_2D import matching, dnn_device

save_path = "/home/hoangqc/Datasets/ZED/fence/"
trajectory = np.loadtxt("/home/hoangqc/Datasets/ZED/fence/pose_left.txt")
pose_mat = quats_to_matrices(trajectory)
pose_mat = pose_mat[::60]

file_rgb = sorted(glob.glob("/home/hoangqc/Datasets/ZED/fence/image/*.jpg"))
file_depth = sorted(glob.glob("/home/hoangqc/Datasets/ZED/fence/depth/*.npy"))

config_cam = {'width':1280, 'height':720,
              'fx':522.5470581054688, 'fy':522.5470581054688,
              'cx':645.46435546875, 'cy':361.37939453125}
cameraIntrinsic = camera.PinholeCameraIntrinsic(**config_cam)

# ros_camExtr = np.array([[0, -1, 0, 0],
#                     [0, 0, -1, 0],
#                     [1, 0, 0, 0],
#                     [0, 0, 0, 1]], dtype=np.float64)

camExtr = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]], dtype=np.float64)

axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

def getFrameInfo(id):
    frame = {   'color': io.imread(file_rgb[id]),
                'depth': np.load(file_depth[id]),
                'transform': pose_mat[id],
                'intr': cameraIntrinsic,
                'extr': camExtr     }
    return frame

id = 30
frame_src = getFrameInfo(id)
id = 33
frame_tar = getFrameInfo(id)

src_cloud_rgb = getPoint3D(frame_src)
tar_cloud_rgb = getPoint3D(frame_tar)

# tar_points = np.asarray(tar_cloud_rgb.points)

K = Cal3_S2(522.5470581054688, 522.5470581054688, 0.0, 645.46435546875, 361.37939453125)
source_cam = PinholeCameraCal3_S2(Pose3(frame_src['transform']), K)
target_cam = PinholeCameraCal3_S2(Pose3(frame_tar['transform']), K)

src_color = frame_src['color']
src_gray = cv.cvtColor(src_color, cv.COLOR_RGB2GRAY)
src_p3d = np.asarray(src_cloud_rgb.points)

frame_tensor = frame2tensor(src_gray, dnn_device)

keys = ['keypoints', 'scores', 'descriptors']
last_data = matching.superpoint({'image': frame_tensor})
last_data = {k + '0': last_data[k] for k in keys}
last_data['image0'] = frame_tensor
last_frame = src_gray

kpts0 = last_data['keypoints0'][0].cpu().numpy()

# for p in kpts0:
#     cv.circle(src_color, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)

kpt0_gt = []
for kp in kpts0:
    x, y = kp
    p0w = Point3(src_p3d[int(y * 1280 + x)])
    q = target_cam.pose().transformTo(p0w)
    pn = target_cam.Project(q)
    pi = target_cam.calibration().uncalibrate(pn)
    kpt0_gt.append(pi.astype(int))

tar_color = frame_tar['color']
gap = 5
prv_img = np.concatenate((src_color, np.full((720,gap,3),dtype=np.uint8, fill_value=255), tar_color), axis=1)
plt.rcParams['figure.dpi'] = 300

for i in range(300):
    x0, y0 = kpts0[i].astype(int)
    x1, y1 = kpt0_gt[i].astype(int)


    if(x1 > 0 and x1 < 1280 and y1 > 0 and y1 < 720):
        cv.drawMarker(prv_img, (x0, y0), (0, 255, 0), cv.MARKER_CROSS, 10, 2)
        cv.drawMarker(prv_img, (x1+1280 + gap, y1), (0, 255, 255), cv.MARKER_SQUARE, 10, 2)

        r = int(np.random.uniform(64, 255))
        g = int(np.random.uniform(64, 255))
        b = int(np.random.uniform(64, 255))
        cv.line(prv_img, (x0, y0), (int(x1 + 1280 + gap), y1), (r, g, b), 1)

plt.imshow(prv_img)
plt.show()


# vis_src_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
# vis_src_cam.transform(frame_src['transform'])
# vis_tar_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
# vis_tar_cam.transform(frame_tar['transform'])
#
# o3d.visualization.draw_geometries([axis_pcd,vis_src_cam,vis_tar_cam,
#                                    src_cloud_rgb.remove_non_finite_points(),
#                                    tar_cloud_rgb.remove_non_finite_points()],
#                                   window_name='pair', width=1024, height=768)