import sys
sys.path.append('../')
from typing import Tuple, Dict

import numpy as np
import open3d as o3d
import cv2 as cv

from os import path
import glob
from functools import partial
from typing import List, Dict, Optional
from dataclasses import dataclass

import gtsam
from gtsam import Point2, Point3, Pose3, Cal3_S2, Values
from gtsam import (Cal3_S2, GenericProjectionFactorCal3_S2,
                   Marginals, NonlinearFactorGraph, PinholeCameraCal3_S2,
                   PriorFactorPose3, PriorFactorPoint3)
from gtsam import symbol_shorthand

from jacobians import cal_Dpose, cal_Dpoint, cal_PointProject_Jacobians

X = symbol_shorthand.X      # Store Poses
L = symbol_shorthand.L      # Store Landmarks
L0 = L(0)
# S = symbol_shorthand.S      # Store Semantics
# kF = symbol_shorthand.F     # Store Frames

@dataclass
class LineSegment3D:
    p0: Point3; p1: Point3

@dataclass
class LineSegment2D:
    p0: Point2; p1: Point2

def error_point_landmarks(  measurement: np.ndarray,
                            calibration: gtsam.Cal3_S2,
                            sems: List,
                            this: gtsam.CustomFactor,
                            values: gtsam.Values,
                            jacobians: Optional[List[np.ndarray]]) -> np.ndarray:
    x_key = this.keys()[0]
    l_key = this.keys()[1]
    # print(sems)
    # s_key = this.keys()[2]

    pose = values.atPose3(x_key)
    pw = values.atPoint3(l_key)
    # conf = values.atDouble(s_key)

    cam = PinholeCameraCal3_S2(pose, calibration)

    # pos = cam.project(point)
    q = cam.pose().transformTo(pw)
    pn = cam.Project(q)
    pi = cam.calibration().uncalibrate(pn)

    error = pi - measurement

    if jacobians is not None:
        # print(gtsam.DefaultKeyFormatter(x_key), gtsam.DefaultKeyFormatter(l_key))
        s = sems[l_key-L0]
        d = 1. / q[2]
        Rt = cam.pose().rotation().transpose()
        Dpose = cal_Dpose(pn, d)
        Dpoint = cal_Dpoint(pn, d, Rt)
        Dpi_pn = np.array([[320., 0], [0, 320.]], dtype=float)
        Dpose = np.matmul(Dpi_pn, Dpose)
        Dpoint = np.matmul(Dpi_pn, Dpoint)
        jacobians[0] = Dpose
        jacobians[1] = Dpoint
        error*=s

    # pos = cam.project(point)
    return error

def createPoses(K: Cal3_S2) -> List[Pose3]:
    """Generate a set of ground-truth camera poses arranged in a circle about the origin."""
    radius = 4.0
    height = -2.0
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    up = gtsam.Point3(0, 0, 1) # NED -> NORTH_EAST_DOWN ???

    target = gtsam.Point3(0, 0, 0)
    poses = []
    for theta in angles:
        position = gtsam.Point3(radius * np.cos(theta), radius * np.sin(theta), height)
        camera = gtsam.PinholeCameraCal3_S2.Lookat(position, target, up, K)
        poses.append(camera.pose())
    return poses

def createPointsLines() -> Tuple[List[Point3],List[LineSegment3D], List[np.array]]:
    step = 5
    points = [
        Point3(-1, -1, 0), Point3(-1, 1, 0),
        Point3(1, -1, 0), Point3(1, 1, 0),
        Point3(-1.5, 1, -1), Point3(1.5, 1, -1),
        Point3(-1.5, -1, -1), Point3(1.5, -1, -1),
        Point3(-1.5, 0, -0.5), Point3(1.5, 0, -0.5)
    ]
    lines = [
        LineSegment3D(points[0], points[1]),
        LineSegment3D(points[2], points[3]),
        LineSegment3D(points[4], points[5]),
        LineSegment3D(points[6], points[7]),
        LineSegment3D(points[8], points[9])
    ]
    semantics = [0.8, 0.8,
                 0.8, 0.8,
                 1.0, 1.0,
                 1.0, 1.0,
                 1.0, 1.0]

    lines_array = []
    for i in range(step):
        l = np.append(points[i*2], points[i*2 + 1])
        lines_array.append(l)

    return [points, lines, semantics]

K = Cal3_S2(320., 320., 0.0, 320., 240.)
pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3])) # RPY - XYZ
measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)  # one pixel in u and v

pl = createPointsLines()
poses = createPoses(K)
points = pl[0]
sems = pl[2]

# Create a factor graph
graph = NonlinearFactorGraph()

#Add Pose Prior
pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))
factor = PriorFactorPose3(X(0), poses[0], pose_noise)
graph.push_back(factor)
# Simulated measurements from each camera pose, adding them to the factor graph
for i, pose in enumerate(poses):
    camera = PinholeCameraCal3_S2(pose, K)
    for j, point in enumerate(points):
        measurement = camera.project(point)
        # print(measurement)
        # factor = GenericProjectionFactorCal3_S2(measurement, measurement_noise, X(i), L(j), K)
        factor = gtsam.CustomFactor(measurement_noise, [X(i),L(j)], partial(error_point_landmarks,measurement, K, sems))
        graph.push_back(factor)

#Add Point Prior
point_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
factor = PriorFactorPoint3(L(0), points[0], point_noise)
graph.push_back(factor)
# graph.print('Factor Graph:\n')

poses_noise = []
points_noise = []
#Store Inital Values
initial_estimate = Values()
for i, pose in enumerate(poses):
    transformed_pose = pose.retract(0.1 * np.random.randn(6, 1))
    poses_noise.append(transformed_pose)
    initial_estimate.insert(X(i), transformed_pose)
for j, point in enumerate(points):
    transformed_point = point + 0.1*np.random.randn(3)
    points_noise.append(transformed_point)
    initial_estimate.insert(L(j), transformed_point)
# initial_estimate.print('Initial Estimates:\n')

# Optimize the graph and print results
params = gtsam.LevenbergMarquardtParams()
params.setVerbosity('TERMINATION')
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
print('Optimizing:')
result = optimizer.optimize()
# result.print('Final results:\n')
print(sems)
print('initial error = {}'.format(graph.error(initial_estimate)))
print('final error = {}'.format(graph.error(result)))


# rs_poses = []
# rs_points = []
# for i, pose in enumerate(poses):
#     rs_poses.append(result.atPose3(X(i)))
# for j, point in enumerate(points_noise):
#     rs_points.append(result.atPoint3(L(j)))
#
# ### VISUALIZATION ###
# from SlamUtils.visualization import getVisualizationBB, getVisualizationPL, getKeyframeGTSAM
#
# vis = o3d.visualization.Visualizer()
# vis.create_window(window_name='Trajectory', width=1024, height=768)
# axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
# vis.add_geometry(axis_pcd)
# line_set = getVisualizationBB(maxZ=3, minZ=-3)
# vis.add_geometry(line_set)
# pointSet = o3d.geometry.PointCloud()
# vis.add_geometry(pointSet)
#
# visL,visP = getVisualizationPL(pl)
# vis.add_geometry(visL)
# vis.add_geometry(visP)
#
# point_error_set = o3d.geometry.PointCloud()
# for p in points_noise:
#     point_error_set.points.append(p)
#     point_error_set.colors.append([1,0,0])
# vis.add_geometry(point_error_set)
#
# point_rs_set = o3d.geometry.PointCloud()
# for p in rs_points:
#     point_rs_set.points.append(p)
#     point_rs_set.colors.append([0,1,0])
# vis.add_geometry(point_rs_set)
#
# for pose in poses:
#     pose_mat = pose.matrix()
#     p = pose.translation()
#     pointSet.points.append(p)
#     # print(pose_mat)
#     camFrame = getKeyframeGTSAM(transform=pose_mat, color=[0, 0, 1])
#     vis.add_geometry(camFrame)
#     vis.update_geometry(pointSet)
#
#     vis.poll_events()
#     vis.update_renderer()
#     ...
#
# for pose in poses_noise:
#     pose_mat = pose.matrix()
#     camFrame = getKeyframeGTSAM(transform=pose_mat, color=[1, 0, 0])
#     vis.add_geometry(camFrame)
#
# for pose in rs_poses:
#     pose_mat = pose.matrix()
#     camFrame = getKeyframeGTSAM(maxX=0.21,minX=-0.21,maxY=0.151, minY=-0.151, transform=pose_mat, color=[0, 1, 0])
#     vis.add_geometry(camFrame)
#
# vis.poll_events();vis.update_renderer();vis.run()
# vis.destroy_window()