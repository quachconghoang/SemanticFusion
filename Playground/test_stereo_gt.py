from pathlib import Path
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

f = 140.278766973
b = 0.3

airsim_path = str(Path.home())+ '/Datasets/Airsim/'

files = ['00000_L.png', '00000_R.png', '00000_D.pfm', '00000_M.png']

l_img = cv.imread(airsim_path+files[0])
r_img = cv.imread(airsim_path+files[1])

depth = cv.imread(airsim_path+files[2], cv.IMREAD_UNCHANGED)
depth = np.flip(depth,0)

l_img_gray = cv.cvtColor(l_img,cv.COLOR_BGR2GRAY)
r_img_gray = cv.cvtColor(r_img,cv.COLOR_BGR2GRAY)

stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(l_img_gray,r_img_gray)
# z = (f*b)/d


orb = cv.ORB_create()
# orb.setFastThreshold(30)
orb.setMaxFeatures(100)
kp1, des1 = orb.detectAndCompute(l_img, None)
kp2, des2 = orb.detectAndCompute(r_img, None)


bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1, des2)
# Sort them in the order of their distance.
matches = sorted(matches, key=lambda x: x.distance)
# Draw first 10 matches.
img_match = cv.drawMatches(l_img, kp1, r_img, kp2, matches[:10],
                      None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


l_img_kp = cv.drawKeypoints(l_img, kp1, None, color=(0,255,0), flags=0)
r_img_kp = cv.drawKeypoints(r_img, kp2, None, color=(0,0,255), flags=0)
img = np.concatenate((l_img_kp, r_img_kp), axis=1)
cv.imshow('***', img)
cv.waitKey()
cv.destroyAllWindows()
