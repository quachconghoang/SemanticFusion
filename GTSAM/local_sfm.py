import sys
sys.path.append('../')
from pathlib import Path
import datetime
import glob
import numpy as np
import cv2 as cv
import open3d as o3d

from SlamUtils.transformation import pos_quats2SEs, pos_quats2SE_matrices, pose2motion, SEs2ses, line2mat, tartan2kitti
from SlamUtils.visualization import getVisualizationBB, getKeyframe
from SlamUtils.utils import dataset_intrinsics
from SlamUtils.Loader.TartanAir import getRootDir, getDataSequences, getDataLists

from Evaluator.evaluator_base import ATEEvaluator, RPEEvaluator, transform_trajs, quats2SEs
import gtsam
from gtsam import Point2, Point3, Rot3, Pose3, Cal3_S2, Values
from gtsam import PinholeCameraCal3_S2

rootDIR = getRootDir()

seq_name = 'office_004'
# seq_name = 'neighborhood_000'
# seq_name = 'seasidetown_000'
save_dir = str(Path.home()) + '/catkin_ws/tmp/sim/' + seq_name + '/'

path = getDataSequences(root=rootDIR, scenario='office', level='Easy', seq_num=4)
# path = getDataSequences(root=rootDIR, scenario='neighborhood', level='Easy', seq_num=0)
# path = getDataSequences(root=rootDIR, scenario='seasidetown', level='Easy', seq_num=0)
files_rgb_left, files_rgb_right, files_depth_left, poselist = getDataLists(dir=path)

# Tartan REF
# poses_mat34 = pos_quats2SEs(poselist)  # [R|t - array 12]
# poses_mat44 = pos_quats2SE_matrices(poselist)
# motions_mat = pose2motion(poses_mat34)  # [R|t]
# motions_quat = SEs2ses(motions_mat).astype(np.float32)  # x-y-z qx-qy-qz-qw

# Load Raw
pose_est = np.loadtxt(fname = save_dir + 'pose_est.txt')[:,:8]
pose_gt = np.loadtxt(fname = save_dir + 'pose_gt.txt')[:,:8]
data_ids = np.loadtxt(fname = save_dir + 'data_id.txt').astype(np.int32)

# Trace start id in estimation
base_time = datetime.datetime(year=2022, month=2, day=22, hour=22, minute=22, second=22).timestamp()
est_time = pose_est[0][0]
start_id = np.round((est_time - base_time)*1000/50).astype(np.int32)

# remove time_stamp
pose_est = pose_est[:,1:]
pose_gt = pose_gt[:,1:]
pose_gt_mat44 = pos_quats2SE_matrices(pose_gt)

# remap to global pose
pose_ref = pose_gt_mat44[start_id-1]
pose_est_mat44 = pos_quats2SE_matrices(pose_est)
for i,pose in enumerate(pose_est_mat44):
    pose_est_mat44[i] = pose_ref.dot(pose)

# remap to estimation to gt
est_ids = [*range(start_id, start_id + pose_est.shape[0])]
pose_gt = pose_gt[est_ids]
pose_gt_mat44 = pos_quats2SE_matrices(pose_gt)

# remap to tartanAIR
data_ids = data_ids[est_ids]
files_rgb_left = [files_rgb_left[i] for i in data_ids]
files_rgb_right = [files_rgb_right[i] for i in data_ids]


# scoring
ate_eval = ATEEvaluator()
rpe_eval = RPEEvaluator()
gt_traj_trans, est_traj_trans, s = transform_trajs(pose_gt, pose_est, cal_scale=False)
gt_SEs, est_SEs = quats2SEs(gt_traj_trans, est_traj_trans)
ate_score, gt_ate_aligned, est_ate_aligned = ate_eval.evaluate(pose_gt, pose_est, scale=False)
rpe_score = rpe_eval.evaluate(gt_SEs, est_SEs)

# rpe_eval.evaluate(gt_SEs[:1000], est_SEs[:1000])
# rpe_eval.evaluate(gt_SEs[1000:2000], est_SEs[1000:2000])
# rpe_eval.evaluate(gt_SEs[2000:3000], est_SEs[2000:3000])
# rpe_eval.evaluate(gt_SEs[3000:3995], est_SEs[3000:3995])

### get transforms -> Keyframe -> rotation thresh -> translation thresh
### VISUALIZATION

vis = o3d.visualization.Visualizer()
vis.create_window(window_name='Trajectory', width=1024, height=768)
axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
vis.add_geometry(axis_pcd)
pose_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25, origin=[0, 0, 0])
line_set = getVisualizationBB()
vis.add_geometry(line_set)
pointSet = o3d.geometry.PointCloud()
vis.add_geometry(pointSet)

num_keyframes = len(pose_est_mat44)

for id in range(num_keyframes):
    camFrame = getKeyframe(transform=pose_est_mat44[id], color=[0,1,0])
    gtCam = getKeyframe(transform=pose_gt_mat44[id], color=[1,1,0])
    vis.add_geometry(camFrame)
    vis.add_geometry(gtCam)
    vis.update_geometry(camFrame);vis.poll_events();vis.update_renderer()

vis.update_geometry(camFrame);vis.poll_events();vis.update_renderer()
vis.run()
vis.destroy_window()