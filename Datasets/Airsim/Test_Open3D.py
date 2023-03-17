import open3d as o3d
import airsim
import numpy as np
from numba import jit, int32, float32
from numba.experimental import jitclass

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


cam = CamParams(IMG_WIDTH, IMG_HEIGHT, IMG_FOV)
xyz = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
