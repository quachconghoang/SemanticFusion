import os.path
from os import path
import glob
import numpy as np
import open3d as o3d
from functools import partial
import cv2 as cv
from skimage import data, io, filters
import sys,shutil
sys.path.append('../')

from SlamUtils.transformation import pos_quats2SEs, pos_quats2SE_matrices, pose2motion, SEs2ses, line2mat, tartan2kitti, quads_NED_to_ENU
from SlamUtils.visualization import getVisualizationBB, getKeyframe
from SlamUtils.utils import dataset_intrinsics
from SlamUtils.Loader.TartanAir import getRootDir, getDataSequences, getDataLists, tartan_camExtr, ros_camExtr

import open3d as o3d
from open3d import camera
from open3d.cuda.pybind.geometry import RGBDImage,PointCloud
from open3d.cuda.pybind.utility import Vector3dVector


rootDIR = getRootDir()
path = getDataSequences(root=rootDIR, scenario='office', level='Easy', seq_num=4)
files_rgb_left, files_rgb_right, files_depth_left, poselist = getDataLists(dir=path)

files_imgs = files_rgb_left[::10]
poses_imgs = poselist[::10]

# from [x, y, z, qx, qy, qz ,qw] to [qw, qx, qy, qz, x, y, z]
poses_colmap = poses_imgs[:,[6,3,4,5,0,1,2]]

rootProject = '/home/hoangqc/COLMAP/test-tartan/'

#copy images to folder - keep the same file name
for i in range(len(files_imgs)):
    src = files_imgs[i]
    dst = rootProject + 'imgs/' + str(i).zfill(6) + '.png'
    shutil.copyfile(src, dst)

file = open(rootProject + 'images.txt','w')
header = "# Image list with two lines of data per image:\n\
# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME"
file.write(header + '\n')
for i in range(poses_colmap.shape[0]):
    data = []
    data.append(str(i+1))
    data.extend(poses_colmap[i].tolist())
    data.append(1)
    fname = str(i).zfill(6) + '.png'
    data.append(fname)
    dumb = ' '.join(map(str,data))
    # print(dumb)
    file.write(dumb + '\n\n')
file.close()

#save numpy
np.savetxt(rootProject + 'poses.txt', poses_colmap)