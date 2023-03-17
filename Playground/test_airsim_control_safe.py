# import setup_path
import airsim

import sys
import time

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

print("arming the drone...")
client.armDisarm(True)
time.sleep(3)

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

z = -4
x = 0
speed = 3 #m/s

print("make sure we are hovering at {} meters...".format(-z))
client.moveToZAsync(z, 1).join()
# client.moveToPositionAsync(x=0,y=0,z=z, velocity=speed)
print("flying on path...")


# z = -5
path_NH_1=[airsim.Vector3r(0,-12, z),
       airsim.Vector3r(0,12, z),
       airsim.Vector3r(0,12, z-4),
       airsim.Vector3r(0,-12, z-4),
        airsim.Vector3r(0, -12, z-8),
        airsim.Vector3r(0, 12, z-8),
            airsim.Vector3r(5, 12, z-8),
            airsim.Vector3r(5, -12, z-8),
            airsim.Vector3r(5, -12, z-4),
            airsim.Vector3r(5, 12, z-4),
            airsim.Vector3r(5, 12, z),
            airsim.Vector3r(5, -12, z),
            airsim.Vector3r(0, 0, z)]


result = client.moveOnPathAsync(path_NH_1, velocity=speed, timeout_sec=600,
        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom, yaw_mode=airsim.YawMode(False,0), lookahead=-1, adaptive_lookahead=1).join()

### MOVE PATHS
# result = client.moveOnPathAsync(path_NH_1, velocity=speed, timeout_sec=600,
#         drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom, yaw_mode=airsim.YawMode(False,0), lookahead=20, adaptive_lookahead=1).join()


# client.moveByVelocityBodyFrameAsync(vx=-1,vy=-1,vz=0,duration=10)
# client.rotateByYawRateAsync(yaw_rate=10, duration=10)
# client.moveByAngleThrottleAsync(pitch=0, roll=0, throttle=0.5, yaw_rate=10, duration=10)

# drone will over-shoot so we bring it back to the start point before landing.

client.moveToPositionAsync(0, 0,-1, velocity=speed).join()
time.sleep(1)
# client.simSetVehiclePose()
# client.moveByVelocityBodyFrameAsync()
print("landing...")
client.landAsync().join()
print("disarming...")
client.armDisarm(False)
client.enableApiControl(False)
print("done.")