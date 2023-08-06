import numpy as np
from scipy.spatial.transform import Rotation as R

from open3d.cuda.pybind.geometry import RGBDImage, PointCloud, Image


def quats_to_matrices(quat_datas):
    data_len = quat_datas.shape[0]
    SEs = []
    for quat in quat_datas:
        SO = R.from_quat(quat[3:7]).as_matrix()
        SE = np.eye(4)
        SE[0:3, 0:3] = SO
        SE[0:3, 3] = quat[0:3]
        SEs.append(SE)
    return SEs

def getPoint3D(frame):
    rgbd0 = RGBDImage.create_from_color_and_depth(
        color=Image(frame['color']),
        depth=Image(frame['depth']),
        depth_scale=1.0, depth_trunc=np.inf,
        convert_rgb_to_intensity=False)
    cloud0 = PointCloud.create_from_rgbd_image(image=rgbd0,
                                               intrinsic=frame['intr'],
                                               extrinsic=frame['extr'],
                                               project_valid_depth_only=False)
    cloud0.transform(frame['transform'])
    return cloud0
    # return np.asarray(cloud0.points)