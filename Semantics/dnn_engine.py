import sys
sys.path.append('../')
from pathlib import Path
import argparse
import random
import numpy as np
import torch
import cv2


from Semantics.SuperGlue.models.matching import Matching
from Semantics.SuperGlue.models.utils import (AverageTimer, VideoStreamer, make_matching_plot_fast, frame2tensor)
from config import dnn_config, dnn_input_size, dnn_device, model_weights

from Semantics.SuperGlue.demo_superpoint import SuperPointFrontend

class DnnEngine():
    def __init__(self):

        self.config = dnn_config
        self.size = dnn_input_size
        self.device = dnn_device

    #     self.fe = SuperPointFrontend(weights_path = model_weights['superpoint'],
    #                             nms_dist = 8,
    #                             conf_thresh = 0.05,
    #                             nn_thresh= 0.7,
    #                             cuda = True)
    #
    #
    #
    # def getSuperPoint(self, img):
    #     pts, desc, heatmap = self.fe.run((img / 255.0).astype('float32'))
    #     kp2D = []
    #     kp2D.clear()
    #     for i in range(pts.shape[1]):
    #         kp2D.append([pts[0][i], pts[1][i]])
    #     return kp2D

