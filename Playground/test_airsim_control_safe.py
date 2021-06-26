# import setup_path
import airsim

import sys
import time

print("""This script is designed to fly on the streets of the Neighborhood environment
and assumes the unreal position of the drone is [160, -1500, 120].""")

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
# client.simEnableWeather(True)
# client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0.25)

p = airsim.Pose(airsim.Vector3r(135, 0, 0), airsim.to_quaternion(0, 0, 0))
client.simSetVehiclePose(pose=p,ignore_collision=True)
time.sleep(1)

print("arming the drone...")
client.armDisarm(True)

state = client.getMultirotorState()
if state.landed_state == airsim.LandedState.Landed:
    print("taking off...")
    client.takeoffAsync().join()
else:
    client.hoverAsync().join()
time.sleep(1)

state = client.getMultirotorState()
if state.landed_state == airsim.LandedState.Landed:
    print("take off failed...")
    sys.exit(1)

# AirSim uses NED coordinates so negative axis is up.
# z of -5 is 5 meters above the original launch point.
z = -5
speed = 5 #m/s

print("make sure we are hovering at {} meters...".format(-z))
client.moveToZAsync(z, 1).join()
# client.moveToPositionAsync(x=0,y=0,z=z, velocity=speed)

# see https://github.com/Microsoft/AirSim/wiki/moveOnPath-demo

# this method is async and we are not waiting for the result since we are passing timeout_sec=0.

print("flying on path...")

path_NH_0=[airsim.Vector3r(125,0,z),
       airsim.Vector3r(125,-130,z),
       airsim.Vector3r(0,-130,z),
       airsim.Vector3r(0,-25,z)]

# path_NH_1=[airsim.Vector3r(125,0,z)]

# path_BD_0=[airsim.Vector3r(0,-8,z),
#            airsim.Vector3r(-8,-8,z),
#            airsim.Vector3r(-8,0,z),
#            airsim.Vector3r(0,0,z)]

result = client.moveOnPathAsync(path_NH_0, velocity=speed, timeout_sec=600,
        drivetrain=airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False,0), lookahead=20, adaptive_lookahead=1).join()

# for i in range(len(path_Park)):
#     client.moveToPositionAsync(path_Park[i].x_val,
#                                path_Park[i].y_val,
#                                path_Park[i].z_val, velocity=speed).join()

# drone will over-shoot so we bring it back to the start point before landing.

client.moveToPositionAsync(0,0,0, velocity=5).join()
# client.simSetVehiclePose()
# client.moveByVelocityBodyFrameAsync()
print("landing...")
client.landAsync().join()
print("disarming...")
client.armDisarm(False)
client.enableApiControl(False)
print("done.")