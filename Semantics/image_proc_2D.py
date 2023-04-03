import cv2 as cv
import numpy as np
from skimage import io
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
import ot

def getImgSobel(img_gray, equalhist = True):
    scale = 1
    delta = 0
    ddepth = cv.CV_16S
    if equalhist:
        img_gray = cv.equalizeHist(img_gray)
    gray = cv.GaussianBlur(img_gray, (3, 3), 0)
    grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)
    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad

def getSobelMask(img_gray, thresh = 30):
    return (getImgSobel(img_gray) > thresh).reshape(-1)

def getLineDistance(pl0,pl1,p):
    return np.linalg.norm(np.cross(pl1 - pl0, pl0 - p)) / np.linalg.norm(pl1 - pl0)

def getLineMinMax(s,err=16):
    min_x=s[0]
    max_x=s[2]
    if s[0]>s[2]:
        min_x = s[2]
        max_x = s[0]

    min_y=s[1]
    max_y=s[3]
    if s[1]>s[3]:
        min_y = s[3]
        max_y = s[1]

    return min_x-err,min_y-err,max_x+err,max_y+err

def showRGB(img):
    plt.imshow(img)
    plt.show()

def showDepth(dimg, dmax = 30):
    plt.imshow(dimg, cmap='gray', vmin=0, vmax=dmax)
    plt.show()

def showNorm(mat, vmax=1.7):
    plt.imshow(mat, cmap='gray', vmin=0, vmax=vmax)
    plt.show()


def getDistanceMask(img_depth, thresh = 15.):
    return (np.abs(img_depth) < thresh).reshape(-1)

import torch
from config import *
from Semantics.SuperGlue.demo_superpoint import SuperPointFrontend
from Semantics.SuperGlue.models.matching import Matching
from Semantics.SuperGlue.models.utils import frame2tensor

torch.set_grad_enabled(False)
# superpointFrontend = SuperPointFrontend(weights_path=model_weights['superpoint'],
#                         nms_dist=8, conf_thresh=0.05,
#                         nn_thresh=0.7, cuda=torch.cuda.is_available())

matching = Matching(dnn_config).eval().to(dnn_device)

def getSuperPoints_v2(src_gray):
    # src_gray_x2 = cv.resize(src_gray, (320, 240), interpolation=cv.INTER_AREA)
    # src_gray_x4 = cv.resize(src_gray, (160, 120), interpolation=cv.INTER_AREA)

    frame_tensor = frame2tensor(src_gray, dnn_device)
    dat = matching.superpoint({'image': frame_tensor})
    # keys = ['keypoints', 'scores', 'descriptors']
    pts = dat['keypoints'][0].cpu().numpy()
    scores = dat['scores'][0].cpu().numpy()
    desc = dat['descriptors'][0].cpu().numpy()
    return {
        'pts': pts,
        'scores': scores,
        'desc': desc
    }

def getSuperPoints_v3(src_gray, scale = 1):
    ...

def hungarianMatch(norm_cross, sorted=False):
    match_id = linear_sum_assignment(cost_matrix = norm_cross)
    match_score = norm_cross[match_id]
    matches = np.stack((match_id[0], match_id[1], match_score), axis=1)

    if sorted:
        idx_sorted = np.argsort(match_score)
        matches = matches[idx_sorted, :]

    return matches

def sinkhornMatch(norm_cross, norm_src, norm_tar, lambd=1e-1, iter=20):
    a, b = [], []
    for pt_his in norm_src:
        a.append(pt_his.mean() - 1)
    for pt_his in norm_tar:
        b.append(pt_his.mean() - 1)

    Gs = ot.sinkhorn(a, b, norm_cross, reg=lambd, numItermax=iter)
    matches = hungarianMatch(1 - Gs)
    return matches

def getAnchorPoints(cost_matrix , his_src, his_tar, sorting = False):
    Gs = ot.sinkhorn(his_src, his_tar, cost_matrix, reg=1e-1, numItermax=20)
    match_id = linear_sum_assignment(cost_matrix = Gs, maximize=True)
    match_score = Gs[match_id]
    matches = np.stack((match_id[0], match_id[1], match_score), axis=1)

    if sorting:
        idx_sorted = np.argsort(-match_score) #high-to-low
        matches = matches[idx_sorted, :]

    theshold = Gs.max()*0.8
    keep = matches[:,2] > theshold
    matches = matches[keep]

    return matches, Gs