import open3d as o3d
import numpy as np
# from numba import jit

def dataset_intrinsics(dataset='tartanair'):
    if dataset == 'kitti':
        focalx, focaly, centerx, centery = 707.0912, 707.0912, 601.8873, 183.1104
    elif dataset == 'euroc':
        focalx, focaly, centerx, centery = 458.6539916992, 457.2959899902, 367.2149963379, 248.3750000000
    elif dataset == 'tartanair':
        focalx, focaly, centerx, centery = 320.0, 320.0, 320.0, 240.0
    else:
        return None
    return focalx, focaly, centerx, centery

# @jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
# def depth2pcd(depthImg, point3D, cx=320, cy=240, fx=320, fy=320, thresh=10.0): # Function is compiled to machine code when called the first time
#     for u in range(depthImg.shape[0]):   # Numba likes loops
#         for v in range(depthImg.shape[1]):
#             d = depthImg[u,v]
#             if abs(d) < thresh:
#                 point3D[u*depthImg.shape[1]+v, :] = [(v-cx)*d/fx, d, -(u-cy)*d/fy ]

def image_to_cloud(depth) -> o3d.geometry.PointCloud():
    cx, cy, f = 320, 240, 320
    pcd = np.zeros(shape=(480 * 640, 3), dtype=np.float32)
    depth2pcd(depth, pcd, cx=cx, cy=cy, fx=f, fy=f)
    rs = o3d.geometry.PointCloud()
    rs.points = o3d.utility.Vector3dVector(pcd)
    return rs
