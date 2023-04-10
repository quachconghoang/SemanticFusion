import sys
import cv2 as cv
import numpy as np
import pyzed.sl as sl

zed = sl.Camera()
# input_path = '/home/hoangqc/Documents/ZED/HD720_SN25607064.svo'
input_path = '/home/hoangqc/Documents/ZED/TestGyro.svo'

init_parameters = sl.InitParameters()
init_parameters.set_from_svo_file(input_path)

# Open the ZED
zed = sl.Camera()
err = zed.open(init_parameters)

svo_image = sl.Mat()
zed.grab()

svo_position = zed.get_svo_position()
zed.retrieve_image(svo_image, sl.VIEW.LEFT_UNRECTIFIED)
cv.imshow("ZED", svo_image.get_data())
key = cv.waitKey(0)
cv.destroyAllWindows()

sensors_data = sl.SensorsData()
# zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE)  # Retrieve only frame synchronized data
# imu_data = sensors_data.get_imu_data()
# linear_acceleration = imu_data.get_linear_acceleration()
# angular_velocity = imu_data.get_angular_velocity()

while zed.grab() == sl.ERROR_CODE.SUCCESS :
    zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE) # Retrieve only frame synchronized data
    # Extract IMU data
    imu_data = sensors_data.get_imu_data()
    # Retrieve linear acceleration and angular velocity
    # linear_acceleration = imu_data.get_linear_acceleration()
    angular_velocity = imu_data.get_angular_velocity()
    print(angular_velocity)


