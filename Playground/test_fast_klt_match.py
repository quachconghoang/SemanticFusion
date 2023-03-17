import cv2 as cv
import numpy as np
import glob
import sys
sys.path.append('../')
from Datasets.bag_loader import RosBagLoader
from config import WIN_EUROC,WIN_AIRSIM

from SuperGlue.demo_superpoint import SuperPointFrontend
import torch
torch.set_grad_enabled(False)

def nms_fast(in_corners, H, W, dist_thresh):
    grid = np.zeros((H, W)).astype(int)  # Track NMS data.
    inds = np.zeros((H, W)).astype(int)  # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2, :])
    corners = in_corners[:, inds1]
    rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
        return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
        out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
        return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
        grid[rcorners[1, i], rcorners[0, i]] = 1
        inds[rcorners[1, i], rcorners[0, i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
        # Account for top and left padding.
        pt = (rc[0] + pad, rc[1] + pad)
        if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
            grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
            grid[pt[1], pt[0]] = -1
            count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid == -1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds

loader = RosBagLoader(config_dict=WIN_EUROC)
loader.loadBag()
fe = SuperPointFrontend(weights_path='./SuperGlue/models/weights/superpoint_v1.pth',
                        nms_dist= 8,
                        conf_thresh=0.03,
                        nn_thresh=0.7,
                        cuda=True)


def detect_fast_grid(src):
    fast = cv.FastFeatureDetector_create()
    fast.setThreshold(15)
    fast.setNonmaxSuppression(12)
    dst = cv.equalizeHist(src)
    kp = fast.detect(dst, None)
    in_corners = np.zeros((3, len(kp)))

    for i in range(len(kp)):
        in_corners[0, i] = kp[i].pt[0]
        in_corners[1, i] = kp[i].pt[1]
        in_corners[2, i] = kp[i].response

    pts, indx = nms_fast(in_corners, 480, 752, dist_thresh=8)  # Apply NMS.
    kp_grid = []
    for i in indx:
        kp_grid.append(kp[i])

    return kp_grid

def detect_superpoint(src):
    input = cv.equalizeHist(src)
    input = input.astype('float32') / 255.0
    pts, desc, heatmap = fe.run(input)
    sp_grid = []
    for i in range(pts.shape[1]):
        sp_grid.append(cv.KeyPoint(x=pts[0, i], y=pts[1, i], size=8, response=pts[2, i]))
    return sp_grid

# index = 1430
l_img, r_img = loader.getImg(1430)
l_img_gray = cv.cvtColor(l_img, cv.COLOR_BGR2GRAY)
r_img_gray = cv.cvtColor(r_img, cv.COLOR_BGR2GRAY)

kp_grid_L = detect_fast_grid(l_img_gray)
# kp_grid_R = detect_fast_grid(r_img_gray)
sp_grid_L = detect_superpoint(l_img_gray)
# sp_grid_R = detect_superpoint(r_img_gray)
# cv.imshow('fast_left',img_L),cv.imshow('fast_right',img_R),cv.waitKey(),cv.destroyAllWindows()

max_kp = 400
img_L = cv.drawKeypoints(l_img, kp_grid_L[:max_kp], None, color=(255,50,50))
# img_R = cv.drawKeypoints(r_img, kp_grid_R[:max_kp], None, color=(255,50,50))
img_SP_L = cv.drawKeypoints(l_img, sp_grid_L[:max_kp], None, color=(0,0,255))
# img_SP_R = cv.drawKeypoints(r_img, sp_grid_R[:max_kp], None, color=(0,0,255))
# cv.imshow('SP_left',img_SP_L),cv.imshow('SP_right',img_SP_R),cv.waitKey(),cv.destroyAllWindows()
# # cv.imshow('fast_left',img_L),cv.imshow('fast_right',img_R),cv.imshow('SP_left',img_SP_L),cv.imshow('SP_right',img_SP_R),cv.waitKey(),cv.destroyAllWindows()
for i in range(max_kp):
    pt_start = (int(kp_grid_L[i].pt[0]),int(kp_grid_L[i].pt[1]))
    cv.circle(img_L, center=pt_start,radius=3,color=(255,0,0),thickness=-1)
for i in range(max_kp):
    pt_start = (int(sp_grid_L[i].pt[0]),int(sp_grid_L[i].pt[1]))
    cv.circle(img_SP_L, center=pt_start,radius=3,color=(0,0,255),thickness=-1)
cv.imshow('fast_left',img_L),cv.imshow('SP_left',img_SP_L),cv.waitKey(),cv.destroyAllWindows()
cv.imwrite('sp-400.png',img_SP_L)
cv.imwrite('fast-400.png',img_L)

lk_params = dict( winSize  = (30, 30), maxLevel = 2, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.01))

p0 = np.ndarray((max_kp,1,2)).astype('float32')
for i in range(max_kp):
    p = kp_grid_L[i]
    p0[i, 0, 0] = p.pt[0]
    p0[i, 0, 1] = p.pt[1]
p1, st, err = cv.calcOpticalFlowPyrLK(l_img_gray, r_img_gray, p0, None, **lk_params)

prev_fast = np.concatenate((img_L, r_img), axis=1)
for i in range(max_kp):
    pt_start = (int(p0[i,0,0]), int(p0[i,0,1]))
    pt_end = (752+int(p1[i,0,0]), int(p1[i,0,1]))
    cv.line(prev_fast,pt1=pt_start,pt2=pt_end,color=(255,0,0))
cv.imshow('prev_fast',prev_fast),cv.waitKey(),cv.destroyAllWindows()

p0 = np.ndarray((max_kp,1,2)).astype('float32')
for i in range(max_kp):
    p = sp_grid_L[i]
    p0[i, 0, 0] = p.pt[0]
    p0[i, 0, 1] = p.pt[1]
p1, st, err = cv.calcOpticalFlowPyrLK(l_img_gray, r_img_gray, p0, None, **lk_params)
prev_SP = np.concatenate((img_SP_L, r_img), axis=1)
for i in range(max_kp):
    pt_start = (int(p0[i,0,0]), int(p0[i,0,1]))
    pt_end = (752+int(p1[i,0,0]), int(p1[i,0,1]))
    cv.line(prev_SP,pt1=pt_start,pt2=pt_end,color=(0,0,255))
cv.imshow('prev_SP',prev_SP),cv.waitKey(),cv.destroyAllWindows()
...

l_img_next, r_img_next = loader.getImg(1435) # 0.3 sec
l_img_next_gray = cv.cvtColor(l_img_next, cv.COLOR_BGR2GRAY)

p0 = np.ndarray((max_kp,1,2)).astype('float32')
for i in range(max_kp):
    p = kp_grid_L[i]
    p0[i, 0, 0] = p.pt[0]
    p0[i, 0, 1] = p.pt[1]
p1, st, err = cv.calcOpticalFlowPyrLK(l_img_gray, l_img_next_gray, p0, None, **lk_params)

prev_fast = np.concatenate((img_L, l_img_next), axis=1)
for i in range(max_kp):
    pt_start = (int(p0[i,0,0]), int(p0[i,0,1]))
    pt_start_next = (752+int(p0[i, 0, 0]), int(p0[i, 0, 1]))
    pt_end = (int(752+p1[i,0,0]), int(p1[i,0,1]))
    cv.line(prev_fast,pt1=pt_start,pt2=pt_end,color=(255,0,0))
    cv.line(prev_fast, pt1=pt_start_next, pt2=pt_end, color=(255, 0, 255), thickness=1)
    cv.circle(prev_fast, center=pt_end, radius=3, color=(255, 0, 255), thickness=1)
cv.imshow('prev_fast',prev_fast),cv.waitKey(),cv.destroyAllWindows()

p0 = np.ndarray((max_kp,1,2)).astype('float32')
for i in range(max_kp):
    p = sp_grid_L[i]
    p0[i, 0, 0] = p.pt[0]
    p0[i, 0, 1] = p.pt[1]
p1, st, err = cv.calcOpticalFlowPyrLK(l_img_gray, l_img_next_gray, p0, None, **lk_params)
prev_SP = np.concatenate((img_SP_L, l_img_next), axis=1)
for i in range(max_kp):
    pt_start = (int(p0[i,0,0]), int(p0[i,0,1]))
    pt_start_next = (752 + int(p0[i, 0, 0]), int(p0[i, 0, 1]))
    pt_end = (int(752+p1[i,0,0]), int(p1[i,0,1]))
    cv.line(prev_SP, pt1=pt_start, pt2=pt_end, color=(0, 0, 255))
    cv.line(prev_SP, pt1=pt_start_next, pt2=pt_end, color=(255, 0, 255), thickness=1)
    cv.circle(prev_SP, center=pt_end, radius=3, color=(255, 0, 255), thickness=1)
cv.imshow('prev_SP',prev_SP),cv.waitKey(),cv.destroyAllWindows()
...