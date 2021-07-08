# import setup_path
import airsim

import sys
import time

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

p = airsim.Pose(airsim.Vector3r(140, -12, -0.2), airsim.to_quaternion(0, 0, 0))
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

z = -3
speed = 3 #m/s

print("make sure we are hovering at {} meters...".format(-z))
# client.moveToZAsync(z, 1).join()
# client.moveToPositionAsync(x=0,y=0,z=z, velocity=speed)
print("flying on path...")

path_NH_0=[airsim.Vector3r(125,0,z),
       airsim.Vector3r(125,-130,z),
       airsim.Vector3r(0,-130,z),
       airsim.Vector3r(0,-25,z)]

path_NH_1=[airsim.Vector3r(140,-12,z),
       airsim.Vector3r(140,12,z),
       airsim.Vector3r(140,12,z-1),
       airsim.Vector3r(140,-12,z-1)]
result = client.moveOnPathAsync(path_NH_1, velocity=speed, timeout_sec=600,
        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom, yaw_mode=airsim.YawMode(False,0), lookahead=-1, adaptive_lookahead=1).join()

z = -5
path_NH_1=[airsim.Vector3r(140,-12,z),
       airsim.Vector3r(140,12,z),
       airsim.Vector3r(140,12,z-1),
       airsim.Vector3r(140,-12,z-1)]
result = client.moveOnPathAsync(path_NH_1, velocity=speed, timeout_sec=600,
        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom, yaw_mode=airsim.YawMode(False,0), lookahead=-1, adaptive_lookahead=1).join()

### MOVE PATHS
# result = client.moveOnPathAsync(path_NH_1, velocity=speed, timeout_sec=600,
#         drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom, yaw_mode=airsim.YawMode(False,0), lookahead=20, adaptive_lookahead=1).join()

# for i in range(len(path_Park)):
#     client.moveToPositionAsync(path_Park[i].x_val,
#                                path_Park[i].y_val,
#                                path_Park[i].z_val, velocity=speed).join()

# drone will over-shoot so we bring it back to the start point before landing.

client.moveToPositionAsync(140,-12,-1, velocity=speed).join()
# client.simSetVehiclePose()
# client.moveByVelocityBodyFrameAsync()
print("landing...")
client.landAsync().join()
print("disarming...")
client.armDisarm(False)
client.enableApiControl(False)
print("done.")