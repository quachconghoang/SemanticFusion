import numpy as np
import open3d as o3d
import cv2 as cv

import sys
sys.path.append('../')

from SlamUtils.transformation import pos_quats2SEs, pos_quats2SE_matrices, pose2motion, SEs2ses, line2mat
# from SlamUtils.utils import image_to_cloud

import open3d as o3d
import numpy as np

T = np.array([[0, 1, 0, 0],
              [0, 0, 1, 0],
              [1, 0, 0, 0],
              [0, 0, 0, 1]], dtype=np.float32)

class Keyframe:
    def __init__(self, rgb, depth, transform=np.eye(4,4), _dataset = 'tartanair', _processing = False):
        self.color_raw = rgb
        self.gray_raw = cv.cvtColor(self.color_raw, cv.COLOR_BGR2GRAY)
        self.depth_raw = depth

        self.pose_mat44 = transform
        # self.extrinsic = np.eye(4, 4)
        self.dataset = _dataset

        self.kp2D = []
        self.kp3D = []
        self.keyCloud = o3d.geometry.PointCloud()
        # self.pcd_blob = image_to_cloud(depth)
        # self.pcd_blob.translate(self.pose_mat44[0:3,3])

        if _processing:
            if self.dataset == 'tartanair':
                self.cameraIntrinsic = o3d.camera.PinholeCameraIntrinsic(width=640, height=480,
                                                                         fx=320, fy=320,
                                                                         cx=320, cy=240)

            self.rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color=o3d.geometry.Image(self.color_raw),
                depth=o3d.geometry.Image(self.depth_raw),
                depth_scale=1.0, depth_trunc=np.inf)
            self.pcd_blob = o3d.geometry.PointCloud.create_from_rgbd_image(image=self.rgbd_image,
                                                                           intrinsic=self.cameraIntrinsic,
                                                                           extrinsic=T)
            self.pcd_blob.transform(self.pose_mat44)
            # print(self.pcd_blob.points)

    def loadKeyCloud(self):
        for i in self.kp2D:
            p = self.pcd_blob.points[int(i[1]*640+i[0])]
            # if (abs(p[1]) < 8.0) & (p[2] > -1.5):
            if (abs(p[1]) < 8.0) :
                self.keyCloud.points.append(p)

    def getData(self, rgb, depth):
        self.color_raw = rgb
        self.gray_raw = cv.cvtColor(self.color_raw, cv.COLOR_RGB2GRAY)
        self.depth_raw = depth
        ...

