import airsim
import cv2
from pathlib import Path

f_c = 140.278766973
c_x = 320
c_y = 200
img_w = 640
img_h = 400
_compress = True
saveDir = str(Path.home())+ '/Datasets/Airsim/'

client = airsim.MultirotorClient()
client.confirmConnection()

v_p = []
v_p.append(airsim.Vector3r(135, 0, -2))
v_p.append(airsim.Vector3r(135.5, 0, -2))
v_p.append(airsim.Vector3r(135.5, 0.5, -2))
v_r = airsim.to_quaternion(0, 0, 0)


if __name__ == "__main__":

    for id in range(3):
        pose = airsim.Pose(v_p[id], v_r)
        client.simSetVehiclePose(pose, ignore_collison=True)
        rLeft = client.simGetImages([
            airsim.ImageRequest("RGB_Left", airsim.ImageType.Scene, compress=_compress),
            airsim.ImageRequest("RGB_Left", airsim.ImageType.DepthPlanner, pixels_as_float=True, compress=_compress),
            airsim.ImageRequest("RGB_Left", airsim.ImageType.Segmentation, compress=_compress)
        ])
        rRight = client.simGetImages([airsim.ImageRequest("RGB_Right", airsim.ImageType.Scene)])

        fname = str(id).zfill(5)
        airsim.write_file((saveDir + fname + '_L.png'), rLeft[0].image_data_uint8)
        airsim.write_file((saveDir + fname + '_R.png'), rRight[0].image_data_uint8)

        airsim.write_pfm(((saveDir + fname + '_D.pfm')), airsim.get_pfm_array(rLeft[1]))
        airsim.write_file((saveDir + fname + '_M.png'), rLeft[2].image_data_uint8)

    airsim.wait_key('Press any key to start!!!')



        # airsim.write_file(os.path.normpath(filename + '.png'), rLeft[0].image_data_uint8)
        # if key == 's':
        #     fname = str(id).zfill(5)
        #     airsim.write_file((saveDir + fname + '_L.png'), rLeft[0].image_data_uint8)
        #     airsim.write_file((saveDir + fname + '_R.png'), rRight[0].image_data_uint8)
        #
        #     airsim.write_pfm(((saveDir + fname + '_D.pfm')), airsim.get_pfm_array(rLeft[1]))
        #     airsim.write_file((saveDir + fname + '_M.png'), rLeft[2].image_data_uint8)
        #     ...

        # if key == 'v':
        #     imgL_0 = cv2.imdecode(airsim.string_to_uint8_array(rLeft[0].image_data_uint8), cv2.IMREAD_UNCHANGED)
        #     imgL_1 = airsim.list_to_2d_float_array(rLeft[1].image_data_float, img_w, img_h)
        #     imgL_2 = cv2.imdecode(airsim.string_to_uint8_array(rLeft[2].image_data_uint8), cv2.IMREAD_UNCHANGED)
        #     imgR_0 = cv2.imdecode(airsim.string_to_uint8_array(rLeft[0].image_data_uint8), cv2.IMREAD_UNCHANGED)


    # cv2.imshow('***', img), cv2.waitKey(), cv2.destroyAllWindows()




