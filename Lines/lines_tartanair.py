import os
import sys
sys.path.append('../')

import numpy as np
import open3d as o3d
import cv2 as cv
from skimage import io

from config import gtsam, Point2, Point3, Rot3, Pose3, Cal3_S2, Values, PinholeCameraCal3_S2
from config import camera, RGBDImage, PointCloud,  Image

from SlamUtils.Loader.TartanAir import getRootDir, getDataSequences, getDataLists, tartan_camExtr, ros_camExtr
from SlamUtils.transformation import pos_quats2SEs, pos_quats2SE_matrices, pose2motion, SEs2ses, line2mat, quads_NED_to_ENU

pcl_room = o3d.io.read_point_cloud('/home/hoangqc/Datasets/TartanAir/office/' + 'office_room.ply')

rootDIR = '/home/hoangqc/Datasets/TartanAir/'
path = getDataSequences(root=rootDIR, scenario='office', level='Easy', seq_num=4)
files_rgb_left, files_rgb_right, files_depth_left, poselist = getDataLists(dir=path, skip=5)
poselist_ENU = np.array([quads_NED_to_ENU(q) for q in poselist])
poses_mat44_ENU = pos_quats2SE_matrices(poselist_ENU)

gtsam_K = Cal3_S2(320, 320, 0.0, 320, 240)
config_cam = {'width':640, 'height':480, 'fx':320, 'fy':320, 'cx':320, 'cy':240}
camIntr = o3d.camera.PinholeCameraIntrinsic(**config_cam) #open3d if need

#----------------------DRAW 3D
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name='Trajectory', width=1024, height=768)
render = vis.get_render_option()
render.point_size = 1.

axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
vis.add_geometry(axis_pcd)
vis.add_geometry(pcl_room)
vis.poll_events();vis.update_renderer()

for id in range(80):
    img_trans = poses_mat44_ENU[id]
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    axis_pcd.transform(img_trans)
    vis.add_geometry(axis_pcd)

    cameraModel = o3d.geometry.LineSet.create_camera_visualization(intrinsic=camIntr,extrinsic=ros_camExtr, scale=0.2)
    cameraModel.colors = o3d.utility.Vector3dVector([[0, 255, 0] for i in range(len(cameraModel.lines))])
    cameraModel.transform(img_trans)
    vis.add_geometry(cameraModel)

    vis.poll_events()
    vis.update_renderer()

vis.poll_events();vis.update_renderer();vis.run()
vis.destroy_window()


def getFrameInfo(id):
    frame = {   'color': io.imread(files_rgb_left[id]),
                'color_right': io.imread(files_rgb_right[id]),
                'depth': np.load(files_depth_left[id]),
                'transform': poses_mat44_ENU[id],
                'intr': camIntr,
                'extr': ros_camExtr     }
    return frame

source = getFrameInfo(32)
target = getFrameInfo(36)
source_cam = PinholeCameraCal3_S2(Pose3(source['transform']), gtsam_K)
target_cam = PinholeCameraCal3_S2(Pose3(target['transform']), gtsam_K)
# src_gray = cv.cvtColor(source['color'], cv.COLOR_RGB2GRAY)
# tar_gray = cv.cvtColor(target['color'], cv.COLOR_RGB2GRAY)

from kornia.feature import SOLD2
import kornia as K
import kornia.feature as KF
import torch

torch_img1 = K.io.load_image(files_rgb_left[32], K.io.ImageLoadType.RGB32)[None, ...]
torch_img2 = K.io.load_image(files_rgb_left[36], K.io.ImageLoadType.RGB32)[None, ...]
torch_img1_gray = K.color.rgb_to_grayscale(torch_img1)
torch_img2_gray = K.color.rgb_to_grayscale(torch_img2)
imgs = torch.cat([torch_img1_gray, torch_img2_gray], dim=0)

sold2 = KF.SOLD2(pretrained=True, config=None)
with torch.inference_mode():
    outputs = sold2(imgs)
line_seg1 = outputs["line_segments"][0]
line_seg2 = outputs["line_segments"][1]
desc1 = outputs["dense_desc"][0]
desc2 = outputs["dense_desc"][1]
with torch.inference_mode():
    matches = sold2.match(line_seg1, line_seg2, desc1[None], desc2[None])

valid_matches = matches != -1
match_indices = matches[valid_matches]
matched_lines1 = line_seg1[valid_matches]
matched_lines2 = line_seg2[match_indices]

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

def plot_images(imgs, titles=None, cmaps="gray", dpi=100, size=6, pad=0.5):
    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n
    figsize = (size * n, size * 3 / 4) if size is not None else None
    fig, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmaps[i]))
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        ax[i].set_axis_off()
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
        if titles:
            ax[i].set_title(titles[i])
    fig.tight_layout(pad=pad)


def plot_lines(lines, line_colors="orange", point_colors="cyan", ps=4, lw=2, indices=(0, 1)):
    if not isinstance(line_colors, list):
        line_colors = [line_colors] * len(lines)
    if not isinstance(point_colors, list):
        point_colors = [point_colors] * len(lines)
    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    axes = [ax[i] for i in indices]
    fig.canvas.draw()

    # Plot the lines and junctions
    for a, l, lc, pc in zip(axes, lines, line_colors, point_colors):
        for i in range(len(l)):
            line = matplotlib.lines.Line2D(
                (l[i, 1, 1], l[i, 0, 1]),
                (l[i, 1, 0], l[i, 0, 0]),
                zorder=1,
                c=lc,
                linewidth=lw,
            )
            a.add_line(line)
        pts = l.reshape(-1, 2)
        a.scatter(pts[:, 1], pts[:, 0], c=pc, s=ps, linewidths=0, zorder=2)


def plot_color_line_matches(lines, lw=2, indices=(0, 1)):
    n_lines = len(lines[0])
    cmap = plt.get_cmap("nipy_spectral", lut=n_lines)
    colors = np.array([mcolors.rgb2hex(cmap(i)) for i in range(cmap.N)])
    np.random.shuffle(colors)
    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    axes = [ax[i] for i in indices]
    fig.canvas.draw()
    # Plot the lines
    for a, l in zip(axes, lines):
        for i in range(len(l)):
            line = matplotlib.lines.Line2D(
                (l[i, 1, 1], l[i, 0, 1]),
                (l[i, 1, 0], l[i, 0, 0]),
                zorder=1,
                c=colors[i],
                linewidth=lw,
            )
            a.add_line(line)

imgs_to_plot = [K.tensor_to_image(torch_img1), K.tensor_to_image(torch_img2)]
lines_to_plot = [line_seg1.numpy(), line_seg2.numpy()]

plot_images(imgs_to_plot, ["Image 1 - detected lines", "Image 2 - detected lines"])
plot_lines(lines_to_plot, ps=3, lw=2, indices={0, 1})