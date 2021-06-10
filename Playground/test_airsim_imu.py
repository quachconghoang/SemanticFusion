import airsim
import cv2
from pathlib import Path
import numpy as np
import json
import gtsam

import rospy
import rosbag
from sensor_msgs.msg import Imu, Image

f_c = 140.278766973
c_x = 320
c_y = 200
img_w = 640
img_h = 400
_compress = True

saveDir = str(Path.home())+ '/Datasets/Airsim/'

client = airsim.MultirotorClient()
client.confirmConnection()
# pose = airsim.Pose(airsim.Vector3r(138, 0, -2), airsim.to_quaternion(0, 0, 0))
# client.simSetVehiclePose(pose, ignore_collison=True)
trajactory = []
id = 0

# imu_data = client.getImuData(imu_name = "IMU", vehicle_name = "CVFlight")

def getImagesResponse(saving=False):
    client.simPause(True)
    responds = client.simGetImages([
        airsim.ImageRequest("RGB_Left", airsim.ImageType.Scene, compress=_compress),
        airsim.ImageRequest("RGB_Left", airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=_compress),
        airsim.ImageRequest("RGB_Left", airsim.ImageType.Segmentation, compress=_compress),
        airsim.ImageRequest("RGB_Right", airsim.ImageType.Scene)
    ])

    if(saving):
        imgL_0 = cv2.imdecode(airsim.string_to_uint8_array(responds[0].image_data_uint8), cv2.IMREAD_UNCHANGED)
        imgL_1 = airsim.list_to_2d_float_array(responds[1].image_data_float, img_w, img_h)
        imgL_2 = cv2.imdecode(airsim.string_to_uint8_array(responds[2].image_data_uint8), cv2.IMREAD_UNCHANGED)
        imgR_0 = cv2.imdecode(airsim.string_to_uint8_array(responds[3].image_data_uint8), cv2.IMREAD_UNCHANGED)

    client.simPause(False)


client.simPause(True)
responds = client.simGetImages([
    airsim.ImageRequest("RGB_Left", airsim.ImageType.Scene, compress=_compress),
    airsim.ImageRequest("RGB_Left", airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=_compress),
    airsim.ImageRequest("RGB_Left", airsim.ImageType.Segmentation, compress=_compress),
    airsim.ImageRequest("RGB_Right", airsim.ImageType.Scene)
])
client.simPause(False)

state = client.getMultirotorState()

imgL_0 = cv2.imdecode(airsim.string_to_uint8_array(responds[0].image_data_uint8), cv2.IMREAD_UNCHANGED)
imgL_1 = airsim.list_to_2d_float_array(responds[1].image_data_float, img_w, img_h)
imgL_2 = cv2.imdecode(airsim.string_to_uint8_array(responds[2].image_data_uint8), cv2.IMREAD_UNCHANGED)
imgR_0 = cv2.imdecode(airsim.string_to_uint8_array(responds[3].image_data_uint8), cv2.IMREAD_UNCHANGED)


while True:

    ...