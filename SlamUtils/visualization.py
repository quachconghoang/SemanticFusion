import os.path
from os import path
import glob
import numpy as np
import open3d as o3d
from typing import Tuple, Dict

# import sys
# sys.path.append('../../')

def getVisualizationBB(maxX=10, maxY=10, maxZ=2, minX=-10, minY=-10, minZ=-2):
    box_points = [[minX, minY, minZ], [maxX, minY, minZ], [minX, maxY, minZ], [maxX, maxY, minZ],
              [minX, minY, maxZ], [maxX, minY, maxZ], [minX, maxY, maxZ], [maxX, maxY, maxZ]]
    box_lines = [[0, 1],[0, 2],[1, 3],[2, 3],[4, 5],[4, 6],[5, 7],[6, 7],[0, 4],[1, 5],[2, 6],[3, 7],]
    box_colors = [[0, 0, 0] for i in range(len(box_lines))]
    line_set = o3d.geometry.LineSet( points=o3d.utility.Vector3dVector(box_points), lines=o3d.utility.Vector2iVector(box_lines))
    line_set.colors = o3d.utility.Vector3dVector(box_colors)

    return line_set

def getVisualizationPL(pl: Tuple):
    points = pl[0]
    point_set = o3d.geometry.PointCloud()
    for p in points:
        point_set.points.append(p)
        point_set.colors.append([0,0,1])

    lines = pl[1]
    vis_point = []
    vis_indices = []
    vis_colors = []
    for i in range(len(lines)):
        ls = lines[i]
        vis_point.append(ls.p0)
        vis_point.append(ls.p1)
        vis_indices.append([i*2, i*2+1])
        vis_colors.append([.2, .2, .8])

    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(vis_point),
                                    lines=o3d.utility.Vector2iVector(vis_indices))
    line_set.colors = o3d.utility.Vector3dVector(vis_colors)
    return line_set, point_set

def getKeyframe(maxX=0.2, maxY=0.2, maxZ=0.15, minX=0.0, minY=-0.2, minZ=-0.15, transform=np.eye(4,4), color = [0,0,1]):
    box_points = [[maxX, minY, minZ], [maxX, minY, maxZ], [maxX, maxY, minZ], [maxX, maxY, maxZ],[0,0,0]]
    box_lines = [[0, 1], [0, 2], [1, 3], [2, 3],
                 [4, 0], [4, 1], [4, 2], [4, 3]]
    box_colors = [color for i in range(len(box_lines))]
    line_set = o3d.geometry.LineSet( points=o3d.utility.Vector3dVector(box_points), lines=o3d.utility.Vector2iVector(box_lines))
    line_set.colors = o3d.utility.Vector3dVector(box_colors)

    return line_set.transform(transform)

def getKeyframeGTSAM(maxX=0.2, maxY=0.15, maxZ=0.2, minX=-0.2, minY=-0.15, minZ=-0.15, transform=np.eye(4,4), color = [0,0,1]):
    box_points = [[maxX, minY, maxZ], [maxX, maxY, maxZ], [minX, minY, maxZ], [minX, maxY, maxZ],[0,0,0]]
    box_lines = [[0, 1], [0, 2], [1, 3], [2, 3],
                 [4, 0], [4, 1], [4, 2], [4, 3]]
    box_colors = [color for i in range(len(box_lines))]
    line_set = o3d.geometry.LineSet( points=o3d.utility.Vector3dVector(box_points), lines=o3d.utility.Vector2iVector(box_lines))
    line_set.colors = o3d.utility.Vector3dVector(box_colors)

    return line_set.transform(transform)

def getErrorSphere():
    ...