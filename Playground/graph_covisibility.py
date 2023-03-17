import numpy as np
import sys
sys.path.append('../')
import numpy as np
import cv2 as cv
import ot
from skimage import io
from scipy.optimize import linear_sum_assignment

from config import camera, RGBDImage, PointCloud, TriangleMesh, Image, Vector3dVector, draw_geometries
from config import gtsam, Point2, Point3, Rot3, Pose3, Cal3_S2, Values, PinholeCameraCal3_S2
from Semantics.image_proc_2D import getImgSobel,getSobelMask, getLineDistance, getLineMinMax,\
                                    showRGB, showDepth, showNorm, getSuperPoints_v2, getAnchorPoints
from Semantics.image_proc_3D import cal_Point_Project_TartanCam, generate3D, \
    getDisparityTartanAIR, get3DLocalFromDisparity, getGroundTruth

from SlamUtils.transformation import pos_quats2SEs, pos_quats2SE_matrices, pose2motion, SEs2ses, line2mat, tartan2kitti
from SlamUtils.visualization import getVisualizationBB, getKeyframe
from SlamUtils.Loader.TartanAir import getRootDir, getDataSequences, getDataLists, tartan_camExtr


rootDIR = getRootDir()
path = getDataSequences(root=rootDIR, scenario='office', level='Easy', seq_num=4)
# path = getDataSequences(root=rootDIR, scenario='neighborhood', level='Easy', seq_num=0) # 104 -> 108 & SKIP 5
files_rgb_left, files_rgb_right, files_depth_left, poselist = getDataLists(dir=path, skip=1)
poses_mat44 = pos_quats2SE_matrices(poselist)

config_cam = {'width':640, 'height':480, 'fx':320, 'fy':320, 'cx':320, 'cy':240}
camIntr = camera.PinholeCameraIntrinsic(**config_cam) #open3d if need
camExtr = tartan_camExtr
K = Cal3_S2(320, 320, 0.0, 320, 240)

def getFrameInfo(id):
    frame = {
        'color': io.imread(files_rgb_left[id]),
        'color_right': io.imread(files_rgb_right[id]),
        'depth': np.load(files_depth_left[id]),
        'transform': poses_mat44[id],
        'intr': camIntr,
        'extr': camExtr
    }
    return frame

def matchSinkhorn(src,tar):
    pts0 = getSuperPoints_v2(cv.cvtColor(src['color'], cv.COLOR_RGB2GRAY))
    pts1 = getSuperPoints_v2(cv.cvtColor(tar['color'], cv.COLOR_RGB2GRAY))
    desc0 = pts0['desc']
    desc1 = pts1['desc']

    norm_self_src = np.sqrt(2 - 2 * np.clip(np.dot(desc0.T, desc0), -1, 1))
    norm_self_tar = np.sqrt(2 - 2 * np.clip(np.dot(desc1.T, desc1), -1, 1))
    norm_cross = np.sqrt(2 - 2 * np.clip(np.dot(desc0.T, desc1), -1, 1))

    src_disparity = getDisparityTartanAIR(src['color'], src['color_right'])

    ...

source = getFrameInfo(160)
generate3D(source)
src_p3d = source['point3D']

source_cam = PinholeCameraCal3_S2(Pose3(source['transform']),K)
src_gray = cv.cvtColor(source['color'], cv.COLOR_RGB2GRAY)

target = getFrameInfo(180)
target_cam = PinholeCameraCal3_S2(Pose3(target['transform']),K)

motion = np.linalg.inv(source['transform']).dot(target['transform'])
motion = Pose3(motion)
rpy = motion.rotation().rpy()
xyz = motion.translation()
rotation_cost = np.linalg.norm(rpy)
translaotion_cost = np.linalg.norm(xyz)