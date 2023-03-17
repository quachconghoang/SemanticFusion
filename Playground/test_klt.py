import cv2 as cv
import numpy as np
import glob
import sys
sys.path.append('../')

import torch
from SuperGlue.demo_superpoint import SuperPointFrontend
from config import model_weights

size = (480, 752, 3)
img_w = 640
img_h = 480
device = 'cuda'


def drawKeyPoints(imgRGB, keypoint, color=(255,0,0)):
    keypoint = keypoint.reshape(keypoint.shape[0], keypoint.shape[2])
    p = tuple(map(tuple, keypoint.astype(int)))
    for pts in p:
        imgRGB = cv.circle(imgRGB, pts, 2, color, thickness=1)
    return imgRGB

def drawMatches(imgRGB, p0, p1, color=(255,0,0)):
    p0 = p0.reshape(p0.shape[0], p0.shape[2])
    p1 = p1.reshape(p1.shape[0], p1.shape[2])
    len = int(p1.shape[0])
    p0 = tuple(map(tuple, p0.astype(int)))
    p1 = tuple(map(tuple, p1.astype(int)))

    for i in range(len):
        pts0=p0[i]
        pts1=p1[i]
        imgRGB = cv.circle(imgRGB, pts1, 2, (255,0,0), thickness=1)
        imgRGB = cv.line(imgRGB,pts0,pts1,color)
        # print(pts0, pts1)
    return imgRGB

def convertImg(input_image):
    rs = cv.cvtColor(input_image, cv.COLOR_RGB2GRAY)
    rs = rs.astype('float32') / 255.0
    return rs

lk_params = dict( winSize  = (20, 20), maxLevel = 3,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.01))

feature_params = dict( maxCorners = 500, qualityLevel = 0.3, minDistance = 7, blockSize = 7 )

print('==> Loading pre-trained network.')
fe = SuperPointFrontend(weights_path=model_weights['superpoint'],
                        nms_dist=4,
                        conf_thresh=0.05,
                        nn_thresh=0.7,
                        cuda=True)
print('==> Successfully loaded pre-trained network.')

path = glob.glob("/home/hoangqc/Datasets/VIODE/seq-4/*.png")
saveDir =  "/home/hoangqc/Datasets/VIODE/rs/"
path.sort()

img0_rgb = cv.imread(path[0])
img1_rgb = cv.imread(path[1])
img_prv = img0_rgb.copy()

img0_rgb = img0_rgb[:,56:56+img_w,:]
img0_gray = convertImg(img0_rgb)
gray0 = cv.cvtColor(img0_rgb, cv.COLOR_RGB2GRAY)

img1_rgb = img1_rgb[:,56:56+img_w,:]
gray1 = cv.cvtColor(img1_rgb, cv.COLOR_RGB2GRAY)

pts, desc, heatmap = fe.run(img0_gray)
pts = pts.transpose()
p0 = np.array([],dtype=np.float32)
for kp in pts:
    p0 = np.append(p0,[np.float32(kp[0]),np.float32(kp[1])])
p0 = p0.reshape((int(p0.shape[0]/2),1,2))

drawKeyPoints(img_prv[:, 56:56 + img_w, :], p0, (0,0,255))
cv.imshow('LK-1', img_prv)
key = cv.waitKey()

# px = cv.goodFeaturesToTrack(img0_gray, mask = None, **feature_params)
# p1, st, err = cv.calcOpticalFlowPyrLK(gray0, gray1, p0, None, **lk_params)
# img_prv = drawKeyPoints(img_prv,p1)

#SWAP
# img0_rgb = img1_rgb
# gray0 = gray1
# p0 = p1
#
# cv.imshow('LK-1', img_prv)
# cv.waitKey()
# cv.destroyAllWindows()

count = 0
name = saveDir + str(count).zfill(5) + '.png'
cv.imwrite(filename=name, img=img_prv)

for f in path:
    img1_rgb = cv.imread(f)
    img_prv = img1_rgb.copy()
    img1_rgb = img1_rgb[:, 56:56 + img_w, :]

    gray1 = cv.cvtColor(img1_rgb, cv.COLOR_BGR2GRAY)
    p1, st, err = cv.calcOpticalFlowPyrLK(gray0, gray1, p0, None, **lk_params)

    drawMatches(img_prv[:, 56:56 + img_w, :], p0=p0, p1=p1, color=(0, 255, 255))
    # img_prv = drawKeyPoints(img_prv[:, 56:56 + img_w, :], p1)

    cv.imshow('LK-1', img_prv)
    key = cv.waitKey()
    if key == 27:
        break

    # SWAP
    img0_rgb = img1_rgb
    gray0 = gray1
    p0 = p1

    count = count+1
    name = saveDir + str(count).zfill(5) + '.png'
    cv.imwrite(filename=name, img=img_prv)


cv.destroyAllWindows()

# cv.imshow('test-1', img_prv), cv.waitKey(), cv.destroyAllWindows()

# detect superpoint
# lucas - kanade flow
# draw something