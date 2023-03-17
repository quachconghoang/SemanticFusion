import os.path
from os import path
import glob
import numpy as np
import open3d as o3d
import cv2 as cv

import sys
sys.path.append('../../')
from Evaluator.transformation import pos_quats2SEs, pose2motion, SEs2ses

_rootDIRS = ['/home/hoangqc/Datasets/TartanAir/',
             '/media/citlab/DATA/TartanAir/']
_scenarios = ['abandonedfactory', 'abandonedfactory_night', 'amusement', 'carwelding',
             'endofworld', 'gascola', 'hospital', 'japanesealley',
             'neighborhood', 'ocean', 'office', 'office2',
             'oldtown', 'seasidetown', 'seasonsforest', 'seasonsforest_winter',
             'soulcity', 'westerndesert']
_levels = ['Easy', 'Hard']

def getRootDir():
    for dir in _rootDIRS:
        if path.exists(dir):
            print("directory exists: ", dir)
            return dir

def getDataSequences(root='', scenario='neighborhood', level='Easy', seq_num=0):
    _scenarios = os.listdir(root)
    path_scr = root
    if any(scenario in s for s in _scenarios):
        path_scr += (scenario + '/' + level + '/')
    else:
        print('loading error at scenario: ', scenario);return None

    _trajs = os.listdir(path_scr)
    _trajs.sort()
    if seq_num < len(_trajs):
        path_scr += (_trajs[seq_num]+'/')
        print('loading path: ',path_scr)
        return path_scr
    else:
        print('loading error at seq: ', seq_num)
        return None

def getDataLists(dir='', skip=1):
    # Load Left-right-GroundTruth
    files_rgb_left = glob.glob(dir + 'image_left/' + '*.png'); files_rgb_left.sort()
    files_rgb_right = glob.glob(dir + 'image_right/' + '*.png'); files_rgb_right.sort()
    files_depth_left = glob.glob(dir + 'depth_left/' + '*.npy'); files_depth_left.sort()
    poselist = np.loadtxt(dir+'pose_left.txt').astype(np.float32)
    if skip > 1:
        return files_rgb_left[0::skip], files_rgb_right[0::skip], files_depth_left[0::skip], poselist[0::skip]
    return files_rgb_left, files_rgb_right, files_depth_left, poselist

def getVisualizationBB(maxX=10, maxY=10, maxZ=2, minX=-10, minY=-10, minZ=-2):
    box_points = [[minX, minY, minZ], [maxX, minY, minZ], [minX, maxY, minZ], [maxX, maxY, minZ],
              [minX, minY, maxZ], [maxX, minY, maxZ], [minX, maxY, maxZ], [maxX, maxY, maxZ]]
    box_lines = [[0, 1],[0, 2],[1, 3],[2, 3],[4, 5],[4, 6],[5, 7],[6, 7],[0, 4],[1, 5],[2, 6],[3, 7],]
    box_colors = [[1, 0, 0] for i in range(len(box_lines))]
    line_set = o3d.geometry.LineSet( points=o3d.utility.Vector3dVector(box_points), lines=o3d.utility.Vector2iVector(box_lines))
    line_set.colors = o3d.utility.Vector3dVector(box_colors)
    return line_set
