import open3d as o3d
import numpy as np
import math
import cv2 as cv
from skspatial.objects import Line, Plane

dirpath = '/home/hoangqc/Datasets/TartanAir/office/'

# pcl_glob = o3d.io.read_point_cloud(dirpath + 'office_glob_1cm_ENU.ply')
# pcl_glob.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=64))
# o3d.io.write_point_cloud(dirpath + 'office_glob_1cm_ENU_normals.ply', pcl_glob, write_ascii=True, compressed=True, print_progress=True)
# point_normals = o3d.io.read_point_cloud(dirpath + 'office_glob_1cm_ENU_normals.ply')

pcl_room = o3d.io.read_point_cloud(dirpath + 'office_room.ply')
axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
point_list = []

def pick_points(pcl):
    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window(window_name='Annotation', width=1024, height=768)
    vis.get_picked_points()
    render = vis.get_render_option()
    render.point_size = 3.
    render.show_coordinate_frame = True
    vis.add_geometry(pcl)
    vis.run()
    vis.destroy_window()
    pp = vis.get_picked_points()
    points = []
    for p in pp:
        points.append(p.coord)
        pcl.colors[p.index] = [255, 0, 0]
    return points


points = np.asarray(pick_points(pcl_room))
point_list.append(points)



dumb_json = []
for pcl in point_list:
    dumb_json.append(pcl.tolist())

import json
with open('annotate.json', 'w') as f:
    json.dump(dumb_json, f)