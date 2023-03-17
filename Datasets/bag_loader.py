import rosbag
from rospy import Time
import datetime
import cv2 as cv
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import pandas as pd
from enum import Enum

class DataType(Enum):
    RGB = 0,
    MONO = 1


class RosBagLoader:
    def __init__(self, config_dict = {}):
        self._info = config_dict
        self.msg = self._info['msg']
        self.img_w = self._info['img_size'][0]
        self.img_h = self._info['img_size'][1]
        self.img_type = self._info['img_type']

        self.folder = self._info['folder']
        self.files = self._info['files']

        self.isTesting = self._info['test_mode']
        self.bagFile = ''
        self._bag_timestamp = []
        if self.isTesting:
            self.bagFile = self._info['test_path']

        self._kind_data = DataType.RGB
        self.img_prv = np.zeros([self.img_h, self.img_w * 2, 3], dtype=np.uint8)
        self.img_0 = np.zeros([self.img_h, self.img_w, 3], dtype=np.uint8)
        self.img_1 = np.zeros([self.img_h, self.img_w, 3], dtype=np.uint8)

        if self.img_type == 'mono16':
            self._kind_data = DataType.MONO
            self.img_prv = np.zeros([self.img_h, self.img_w * 2], dtype=np.uint16)
            self.img_0 = np.zeros([self.img_h, self.img_w], dtype=np.uint16)
            self.img_1 = np.zeros([self.img_h, self.img_w], dtype=np.uint16)

        ...

    def loadBag(self):
        self.bridge = CvBridge()
        self._bag = rosbag.Bag(self.bagFile)
        self._img_count = self._bag.get_message_count(topic_filters=self.msg[0])

        self._bag_timestamp.clear()
        for topic, msg, t in self._bag.read_messages(topics=[self.msg[0]]):
            self._bag_timestamp.append(t)
        # self.bagpath = _Path
        print('Img count = ', self._img_count)
        ...

    def getImg(self, index=0):
        i = index
        st = Time(self._bag_timestamp[i].secs, self._bag_timestamp[i].nsecs - 1e6)
        if i > 0: st = Time(self._bag_timestamp[i - 1].secs, self._bag_timestamp[i - 1].nsecs + 1e6)
        et = Time(self._bag_timestamp[i].secs, self._bag_timestamp[i].nsecs)
        for topic, msg, t in self._bag.read_messages(topics=self.msg, start_time=st, end_time=et):
            if topic == self.msg[0]:
                self.img_0 = self.bridge.imgmsg_to_cv2(msg, self.img_type)
                # self.img_prv[:, :self.img_w] = self.bridge.imgmsg_to_cv2(msg, self.img_type)
            if topic == self.msg[1]:
                self.img_1 = self.bridge.imgmsg_to_cv2(msg, self.img_type)
                # self.img_prv[:, self.img_w:(self.img_w * 2)] = self.bridge.imgmsg_to_cv2(msg, self.img_type)
            # if topic == '/odometry':
            #     odom = msg

        return self.img_0, self.img_1

    def getImgCount(self):
        return self._img_count

    def loadTrajactory(self):
        ...
    ...