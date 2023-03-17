import numpy as np
from gtsam import Point2, Point3, Pose3, Cal3_S2, PinholeCameraCal3_S2
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class Line3D:
    p0: Point3; p1: Point3

@dataclass
class Line2D:
    p0: Point2; p1: Point2

@dataclass
class KeyFrame:
    keypoints: List[Point2];
    keylines: List[Line2D]

def createPointsLines() -> Tuple[List[Point3], List[Line3D]]:
    points = [
        Point3(-1, -1, 0), Point3(-1, 1, 0),
        Point3(1, -1, 0), Point3(1, 1, 0),
        Point3(-1.5, 1, -1), Point3(1.5, 1, -1),
        Point3(-1.5, -1, -1), Point3(1.5, -1, -1),
        Point3(-1.5, 0, -0.5), Point3(1.5, 0, -0.5)
    ]
    lines = [Line3D(points[0], points[1]), Line3D(points[0], points[1]),
            Line3D(points[2], points[3]), Line3D(points[2], points[3]),
            Line3D(points[4], points[5]), Line3D(points[4], points[5]),
            Line3D(points[6], points[7]), Line3D(points[6], points[7]),
            Line3D(points[8], points[9]), Line3D(points[8], points[9]),]

    return [points, lines]

def createPoses(K: Cal3_S2) -> List[Pose3]:
    """Generate a set of ground-truth camera poses arranged in a circle about the origin."""
    radius = 4.0
    height = -2.0
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    up = Point3(0, 0, 1) # NED -> NORTH_EAST_DOWN ???

    target = Point3(0, 0, 0)
    poses = []
    for theta in angles:
        position = Point3(radius * np.cos(theta), radius * np.sin(theta), height)
        camera = PinholeCameraCal3_S2.Lookat(position, target, up, K)
        poses.append(camera.pose())
    return poses

def fixError():
    poseErr = []
    poseErr.append(np.array([[ 1.04109795],[ 0.34445191],[-1.72685288],[-1.07914509],[ 0.26222813],[ 0.18995801]]))# 0
    poseErr.append(np.array([[-1.56685498],[-0.31235944],[ 1.83844975],[ 0.3493736 ],[-0.48584567],[-0.56244297]]))# 1
    poseErr.append(np.array([[-2.70874461],[ 0.23451095],[ 0.95209185],[ 0.8790236 ],[ 0.07261368],[-0.73301408]]))# 2
    poseErr.append(np.array([[ 0.66462284],[-0.46475181],[-0.39577424],[ 0.35666478],[-0.0773355 ],[-0.66910307]]))# 3

    poseErr.append(np.array([[ 0.40594707],[-2.28276338],[-0.16590538],[ 0.36279499],[-1.40064523],[ 1.65829899]]))# 4
    poseErr.append(np.array([[-0.60210771],[ 1.46278761],[ 1.16374325],[-2.26078036],[ 0.29489038],[ 0.4970514 ]]))# 5
    poseErr.append(np.array([[-0.50983245],[ 0.28197948],[-0.37941486],[ 0.28638089],[-0.11355351],[ 0.3036549 ]]))# 6
    poseErr.append(np.array([[ 0.06336537],[ 0.22699721],[-2.14409641],[-0.73139963],[-2.1267864 ],[ 0.03086451]]))# 7

    pointErr = []
    pointErr.append(np.array([ 1.72222411,  0.53795576,  0.77165312]))#0
    pointErr.append(np.array([ 1.29701603, -1.18109136, -1.61784449]))
    pointErr.append(np.array([ 0.65463539, -1.24916863,  0.16089417]))
    pointErr.append(np.array([-1.12778816,  1.42504328,  0.99567022]))
    pointErr.append(np.array([ 0.87451795, -0.3584813 , -0.10962744]))

    pointErr.append(np.array([ 1.0573157 , -0.45491569,  0.72048178]))#5
    pointErr.append(np.array([ 0.75653916,  1.39721628, -0.07245901]))
    pointErr.append(np.array([-0.36571246,  0.28479449,  0.43267804]))
    pointErr.append(np.array([ 0.94985898, -1.98758551, -0.32677346]))
    pointErr.append(np.array([ 2.21849903, -0.05167009, -0.16495746]))

    return poseErr,pointErr