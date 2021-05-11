import sys
sys.path.append('../')
from Datasets.bag_loader import RosBagLoader
from config import TUMVI, VIODE
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # print(VIODE)
    loader = RosBagLoader(config_dict=VIODE)
    loader.loadBag()

    # for i in range(loader.getImgCount()):
    #     l_img, r_img, odom = loader.getImg(i)
    #     img = np.concatenate((l_img, r_img), axis=1)
    #     print(i, odom.position)
    #     cv.imshow('***', img)
    #     k = cv.waitKey(1)
    #     if k == 27:
    #         break
    # cv.destroyAllWindows()

    l_img, r_img, odom = loader.getImg(600)
    # Initiate ORB detector
    orb = cv.ORB_create()
    # orb.setFastThreshold(30)
    orb.setMaxFeatures(100)
    kp1, des1 = orb.detectAndCompute(l_img, None)
    kp2, des2 = orb.detectAndCompute(r_img, None)

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    # Draw first 10 matches.
    img3 = cv.drawMatches(l_img, kp1, r_img, kp2, matches[:1],
                          None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # plt.imshow(img3), plt.show()
    # l_img = cv.drawKeypoints(l_img, kp1, None, color=(0,255,0), flags=0)
    # r_img = cv.drawKeypoints(r_img, kp2, None, color=(0,0,255), flags=0)
    # img = np.concatenate((l_img, r_img), axis=1)

    cv.imshow('***', img3)
    cv.waitKey()
    cv.destroyAllWindows()

    # https://towardsdatascience.com/depth-estimation-1-basics-and-intuition-86f2c9538cd1
    # z = = (f*b)/d