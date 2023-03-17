import gtsam
from gtsam import Point2, Point3, Pose3, Cal3_S2, Values
from gtsam import (Cal3_S2, PinholeCameraCal3_S2)
from gtsam import symbol_shorthand
import numpy as np

def cal_Dpose(  pn: Point2,
                d: np.float64) -> np.ndarray:
    u,v = pn[0], pn[1]
    uv = u*v;    uu = u*u;    vv = v*v
    Dpn_pose = np.array([[uv, (-1 -uu), v, -d, 0, d*u],
                        [1 + vv, -uv, -v, 0, -d, d*v]], dtype=np.float64);
    return Dpn_pose

def cal_Dpoint(pn: Point2,
                d: np.float64,
               Rt: np.array) -> np.ndarray:
    u,v = pn[0], pn [1]
    Dpn_point = np.array([[Rt[0,0] - u*Rt[2,0], Rt[0,1] - u*Rt[2,1], Rt[0,2] - u*Rt[2,2]],
                         [Rt[1,0] - v*Rt[2,0], Rt[1,1] - v*Rt[2,1], Rt[1,2] - v*Rt[2,2]]], dtype=np.float64);
    Dpn_point*=d
    return Dpn_point

def cal_PointProject_Jacobians(cam: PinholeCameraCal3_S2,
                                pw: Point3):
    q = cam.pose().transformTo(pw)
    pn = cam.Project(q)
    pi = cam.calibration().uncalibrate(pn)
    d = 1 / q[2]
    Rt = cam.pose().rotation().transpose()
    Dpose = cal_Dpose(pn, d)
    Dpoint = cal_Dpoint(pn, d, Rt)
    Dpi_pn = np.array([[cam.calibration().fx(), 0], [0, cam.calibration().fy()]])
    Dpose = np.matmul(Dpi_pn, Dpose)  # - ?
    Dpoint = np.matmul(Dpi_pn, Dpoint)  # - ?
    return pi, Dpose, Dpoint

def point_on_line(a, b, p):
    ap = p - a
    ab = b - a
    result = a + np.dot(ap, ab) / np.dot(ab, ab) * ab
    return result


# p = Point2(1,1)
#
# a = Point2(1,1)
# b = Point2(-1,-1)
# dis = point_on_line(a,b, p)

K = Cal3_S2(320, 320, 0.0, 320, 240)
target = gtsam.Point3(0, 0, 0)
position = gtsam.Point3(20,0,0)
up = gtsam.Point3(0, 0, 1) # NED -> NORTH_EAST_DOWN ???

cam = gtsam.PinholeCameraCal3_S2.Lookat(position, target, up, K)
pw = Point3(10.0, 3.0, 0.0)

# pi_groundtruth = cam.project(pw)
# pi,Dpose,Dpoint = cal_PointProject_Jacobians(cam,pw)


q = cam.pose().transformTo(pw)
pn = cam.Project(q)
pi = cam.calibration().uncalibrate(pn)
d = 1/q[2]
#
# Rt = cam.pose().rotation().transpose()
# H1 = cal_Dpose(pn,d)
# H2 = cal_Dpoint(pn,d,Rt)
#
# Dpi_pn = np.array([[320,0], [0,320]])
# H1 = np.matmul(Dpi_pn,H1) # - ?
# H2 = np.matmul(Dpi_pn,H2) # - ?