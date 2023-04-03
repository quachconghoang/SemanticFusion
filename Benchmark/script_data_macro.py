import glob
from zipfile import ZipFile, BadZipFile
import os

download_path = '/media/hoangqc/Expansion/Datasets/tartanair_tools/Downloads/'
unzip_path = '/media/hoangqc/Expansion/Datasets/TartanAir/'

# download_path = 'D:/Datasets/tartanair_tools/Downloads/'
# unzip_path = 'D:/Datasets/TartanAir/'

TartanAir_scenarios = ['abandonedfactory', 'abandonedfactory_night', 'amusement', 'carwelding',
             'endofworld', 'gascola', 'hospital', 'japanesealley',
             'neighborhood', 'ocean', 'office', 'office2',
             'oldtown', 'seasidetown', 'seasonsforest', 'seasonsforest_winter',
             'soulcity', 'westerndesert']

files_img_left = []
for sce in TartanAir_scenarios:
    img_ez= os.path.join(download_path, sce,'Easy','image_left.zip')
    img_hd = os.path.join(download_path, sce, 'Hard', 'image_left.zip')
    if os.path.isdir(os.path.join(unzip_path,'image_left',sce,sce,'Easy')) == False:
        files_img_left.append(img_ez)
    if os.path.isdir(os.path.join(unzip_path,'image_left',sce,sce,'Hard')) == False:
        files_img_left.append(img_hd)

files_img_right = []
for sce in TartanAir_scenarios:
    img_ez = os.path.join(download_path, sce,'Easy','image_right.zip')
    img_hd = os.path.join(download_path, sce, 'Hard', 'image_right.zip')
    if os.path.isdir(os.path.join(unzip_path,'image_right',sce,sce,'Easy')) == False:
        files_img_right.append(img_ez)
    if os.path.isdir(os.path.join(unzip_path,'image_right',sce,sce,'Hard')) == False:
        files_img_right.append(img_hd)

files_depth_left = []
for sce in TartanAir_scenarios:
    img_ez = os.path.join(download_path, sce,'Easy','depth_left.zip')
    img_hd = os.path.join(download_path, sce, 'Hard', 'depth_left.zip')
    if os.path.isdir(os.path.join(unzip_path,'depth_left',sce,sce,'Easy')) == False:
        files_depth_left.append(img_ez)
    if os.path.isdir(os.path.join(unzip_path,'depth_left',sce,sce,'Hard')) == False:
        files_depth_left.append(img_hd)

print('Print rgb-left ...')
for file in files_img_left:
    with ZipFile(file, 'r') as zObject:
        zObject.extractall(unzip_path+'image_left/')
    print(file)

print('Print rgb-right ...')
for file in files_img_right:
    with ZipFile(file, 'r') as zObject:
        zObject.extractall(unzip_path+'image_right/')
    print(file)

print('Print depth-left ...')
for file in files_depth_left:
    with ZipFile(file, 'r') as zObject:
        zObject.extractall(unzip_path+'depth_left/')
    print(file)