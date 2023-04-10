from pathlib import Path
import datetime
from sys import platform

import open3d
from open3d import camera
from open3d import _build_config as open3d_build_config
if (open3d.__DEVICE_API__ == 'cuda'):
    from open3d.cuda.pybind.geometry import RGBDImage,PointCloud, TriangleMesh, Image
    from open3d.cuda.pybind.utility import Vector3dVector
    from open3d.cuda.pybind.visualization import draw_geometries
else:
    from open3d.cpu.pybind.geometry import RGBDImage,PointCloud, TriangleMesh, Image
    from open3d.cpu.pybind.utility import Vector3dVector
    from open3d.cpu.pybind.visualization import draw_geometries

# from open3d import geometry

import gtsam
from gtsam import Point2, Point3, Rot3, Pose3, Cal3_S2, Values
from gtsam import PinholeCameraCal3_S2


if (platform =='linux') | (platform =='darwin'):
    model_weights = {
        'superpoint': str(Path.home()) + '/Datasets/Weights/superpoint_v1.pth',
        'superglue_indoor': str(Path.home()) + '/Datasets/Weights/superglue_indoor.pth',
        'superglue_outdoor': str(Path.home()) + '/Datasets/Weights/superglue_outdoor.pth',
    }
else:
    model_weights = {
        'superpoint': 'C:/Users/hoangqc/Desktop/ws/weights/superpoint_v1.pth',
        'superglue_indoor': 'C:/Users/hoangqc/Desktop/ws/weights/superglue_indoor.pth',
        'superglue_outdoor': 'C:/Users/hoangqc/Desktop/ws/weights/superglue_outdoor.pth',
    }

dnn_config = {
    'superpoint': {
        'nms_radius': 8,
        'keypoint_threshold': 0.05,
        # 'keypoint_threshold': 0.005,
        'max_keypoints': 1024
    },
    # 'superpoint': {
    #     'nms_radius': 4,
    #     'keypoint_threshold': 0.01,
    #     'max_keypoints': 2048
    # },
    'superglue': {
        'weights': 'outdoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
    }
}

dnn_input_size = (640,480)
dnn_device = 'cuda'
if platform == 'darwin':
    dnn_device = 'cpu'

TartanAir_rootDIRS = [  '/home/hoangqc/Datasets/TartanAir/',
                        '/media/hoangqc/DATA/TartanAir/',
                        '/media/hoangqc/Expansion/Datasets/TartanAir/',
                        '...'   ]

TartanAir_scenarios = ['abandonedfactory', 'abandonedfactory_night', 'amusement', 'carwelding',
             'endofworld', 'gascola', 'hospital', 'japanesealley',
             'neighborhood', 'ocean', 'office', 'office2',
             'oldtown', 'seasidetown', 'seasonsforest', 'seasonsforest_winter',
             'soulcity', 'westerndesert']

TartanAir_levels = ['Easy', 'Hard']

TartanAir_bag_st = datetime.datetime(year=2023, month=3, day=23, hour=23, minute=23, second=23).timestamp()
