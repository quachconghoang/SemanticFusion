import sys
sys.path.append('../')
from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch
import cv2


from SuperGlue.models.matching import Matching
from SuperGlue.models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)

torch.set_grad_enabled(False)
config = {
        'superpoint': {
            'nms_radius': 8,
            'keypoint_threshold': 0.005,
            'max_keypoints': -1
        },
        'superglue': {
            'weights': 'outdoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
}
device = 'cuda'
size = (640, 480)
blank_image = np.zeros((480,10,3), np.uint8)
blank_image.fill(255)

def convertImg(_img):
    tmp = cv2.cvtColor(_img, cv2.COLOR_RGB2GRAY)
    tmp = cv2.resize(tmp,size)
    return tmp


matching = Matching(config).eval().to(device)
keys = ['keypoints', 'scores', 'descriptors']

airsim_path = str(Path.home())+ '/Datasets/Airsim/Random-NH/'
files = ['00007_L.png', '00000_R.png', '00000_D.pfm', '00000_M.png','00008_L.png']
passThres = 0.9

img_0 = cv2.imread(airsim_path+files[0])
# img_1 = cv2.imread(airsim_path+files[1])
img_2 = cv2.imread(airsim_path+files[0+4])
img_raw = np.concatenate((img_0, blank_image, img_2), axis=1)


# FRAME 0
frame = convertImg(img_0)
frame_tensor = frame2tensor(frame, device)
last_data = matching.superpoint({'image': frame_tensor})
last_data = {k+'0': last_data[k] for k in keys}
last_data['image0'] = frame_tensor
last_frame = frame

# FRAME 1
frame = convertImg(img_2)
frame_tensor = frame2tensor(frame, device)
pred = matching({**last_data, 'image1': frame_tensor})
kpts0 = last_data['keypoints0'][0].cpu().numpy()
kpts1 = pred['keypoints1'][0].cpu().numpy()
matches = pred['matches0'][0].cpu().numpy()
confidence = pred['matching_scores0'][0].cpu().numpy()

valid = matches > -1
mkpts0 = kpts0[valid]
mkpts1 = kpts1[matches[valid]]
color = cm.jet(confidence[valid])

cf = confidence[valid] > passThres
mkpts0_r = mkpts0[cf]
mkpts1_r = mkpts1[cf]
color_r = color[cf]

text = [
    # 'SuperGlue',
    # 'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
    # 'Matches: {}'.format(len(mkpts0))
]
k_thresh = matching.superpoint.config['keypoint_threshold']
m_thresh = matching.superglue.config['match_threshold']
# out = make_matching_plot_fast(
#     last_frame, frame, kpts0, kpts1, mkpts0, mkpts1, color, text,
#     path=None, show_keypoints=True)
out = make_matching_plot_fast(
    last_frame, frame, kpts0, kpts1, mkpts0_r, mkpts1_r, color_r, text,
    path=None, show_keypoints=True)

#cv2.imshow('abc',out),cv2.waitKey(),cv2.destroyAllWindows()
cv2.imshow('match',out)
cv2.imshow('raw',img_raw)
cv2.waitKey()
cv2.destroyAllWindows()