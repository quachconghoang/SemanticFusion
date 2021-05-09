import open3d as o3d

if __name__ == "__main__":
    # print(VIODE)
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(axis_pcd)
    pointSet = o3d.geometry.PointCloud()
    vis.add_geometry(pointSet)

    p = [0,0,0]
    pointSet.points.append(p)
    vis.update_geometry(pointSet)
    vis.poll_events()
    vis.update_renderer()