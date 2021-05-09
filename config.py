from pathlib import Path

VIODE = {
    'msg': ['/cam0/image_raw', '/cam1/image_raw', '/cam0/segmentation', '/cam1/segmentation',
             '/imu0', '/odometry'],
    'img_size': [752,480],
    'img_type' : 'bgr8',
    'folder': str(Path.home())+ '/Datasets/VIODE/city_day/',
    'files': ['0_none.bag', '1_low.bag', '2_mid.bag', '3_high.bag'],
    'test_path': str(Path.home())+ '/Datasets/VIODE/city_day/0_none.bag',
    'test_mode': True
}

OpenLORIS = {

}

TUMVI = {
    'msg': ['/cam0/image_raw', '/cam1/image_raw',
           '/imu0', '/vrpn_client/raw_transform'],
    'img_size': [512,512],
    'img_type' : 'mono16',
    'folder': str(Path.home())+ '/Datasets/TUM_VI/',
    'files': ['dataset-corridor1_512_16.bag'],
    'test_path': str(Path.home())+ '/Datasets/TUM_VI/dataset-corridor1_512_16.bag',
    'test_mode': True
}

