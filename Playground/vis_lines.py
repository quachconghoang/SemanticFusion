import numpy as np
import open3d as o3d

file_lines = "/home/hoangqc/Datasets/Hilti-2022/construction_lines.txt"
file_ply = "/home/hoangqc/Datasets/Hilti-2022/construction_site_upper_level_1cm.ply"
lines = np.loadtxt(file_lines, delimiter=None)

xyz = lines[:, 0:3]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

pcd_raw = o3d.io.read_point_cloud(file_ply)
o3d.visualization.draw_geometries([pcd,pcd_raw])
