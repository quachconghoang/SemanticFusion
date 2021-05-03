import sys
sys.path.append('../')
from Datasets.bag_loader import RosBagLoader
from config import TUMVI, VIODE
import numpy as np
import cv2 as cv

if __name__ == "__main__":
    # print(VIODE)
    loader = RosBagLoader(config_dict=VIODE)
    loader.loadBag()
    l_img, r_img, odom = loader.getImg(100)

#FAST
    # fast = cv.FastFeatureDetector_create()
    # kp = fast.detect(l_img, None)
    # img2 = cv.drawKeypoints(l_img, kp, None, color=(255, 0, 0))
    # # Print all default params
    # print("Threshold: {}".format(fast.getThreshold()))
    # print("nonmaxSuppression:{}".format(fast.getNonmaxSuppression()))
    # print("neighborhood: {}".format(fast.getType()))
    # print("Total Keypoints with nonmaxSuppression: {}".format(len(kp)))

#ORB

    # Initiate ORB detector
    orb = cv.ORB_create()
    kp = orb.detect(l_img,None)
    kp, des = orb.compute(l_img, kp)
    img2 = cv.drawKeypoints(l_img, kp, None, color=(0,255,0), flags=0)

    # img = np.concatenate((l_img, r_img), axis=1)
    # cv.imshow('***', img2)
    # cv.waitKey()
    # cv.destroyAllWindows()

    # loader