from open3d import *
import airsim
import numpy as np
from numba import jit, jitclass, int32, float32
from timeit import default_timer as timer

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_FOV = 135 #degree

#C/C++ Struct Style
spec = [
    ('cx', float32),
    ('cy', float32),
    ('fx', float32),
    ('fy', float32)
]

@jitclass(spec)
class CamParams:
    def __init__(self, width, height, fov):
        self.cx = np.float32((width-1)/2)
        self.cy = np.float32((height-1)/2)
        self.fx = np.float32( (width/2) / np.tan(np.deg2rad(fov)/2) )
        self.fy = np.float32( (height/2) / np.tan(np.deg2rad(fov)/2) )

# High Performance Kernel ~ C/C++
#@jit(nopython=True, fastmath=True, parallel=True)
@jit
def depth2xyz(cam, image, xyz):
    for i, j in np.ndindex(image.shape):
        depth = image[i,j]
        xyz[i, j, 0] = (i - cam.cx) * depth / cam.fx
        xyz[i, j, 1] = (j - cam.cy) * depth / cam.fy
        xyz[i, j, 2] = depth


depth_airsim, scale = airsim.read_pfm("./TestData/testAirSim.pfm")

cam = CamParams(IMG_WIDTH, IMG_HEIGHT, IMG_FOV)
xyz = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
depth2xyz(cam, depth_airsim, xyz)

#s = timer()
#for i in range(500):
#    img2cloud(cam, depth_airsim, cloud)
#e = timer()
#print((e-s)/500)

pcd = PointCloud()
pcd.points = Vector3dVector(xyz.reshape(IMG_WIDTH*IMG_HEIGHT,3))
draw_geometries([pcd, create_mesh_coordinate_frame(size = 5, origin = [0, 0, 0])])