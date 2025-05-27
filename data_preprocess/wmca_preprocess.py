import h5py
import cv2
import numpy as np
from PIL import Image
import os


def extract_hdf5(h5_path, out_dir, modal='color'):
    h5_dataset = h5py.File(h5_path, 'r')
    # print(h5_dataset.keys())
    for key in h5_dataset.keys():
        # key = 'Frame_{}'.format(i)
        color_data = h5_dataset[key]['array'][:]  # (640, 830, 3) unit8
        if modal == 'color':
            img = color_data
        elif modal == 'gray':
            img = color_data[0]
        elif modal == 'depth':
            img = color_data[1]
        elif modal == 'ir':
            img = color_data[2]
        elif modal == 'thermal':
            img = color_data[3]
        if modal == 'color':
            img = np.transpose(img, (1, 2, 0))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # else:
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        i = int(key.replace('Frame_', ''))
        cv2.imwrite(f'{out_dir}/{modal}/{i:04}.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        # print(f'{out_dir}/{modal}/{i:04}.jpg')

        # break


h5_rgb_folder = "/home/hdd1/share/public_data/FAS-2023/WMCA/WMCA_preprocessed_RGB/WMCA/face-station"
h5_cdit_folder = "/home/hdd1/share/public_data/FAS-2023/WMCA/WMCA_preprocessed_CDIT/WMCA/face-station"
out = "/home/hdd1/share/public_data/FAS-2023/WMCA/image"

for sub_folder in os.listdir(h5_rgb_folder):
    whole_folder = os.path.join(h5_rgb_folder, sub_folder)
    if sub_folder == '12.02.18':
        print('good !!!!')
    for h5_file in os.listdir(whole_folder):
        if h5_file == '100_03_015_2_10.hdf5':
            print('find it !!!!')
        out_h5_folder = os.path.join(out, h5_file.replace('.hdf5', '').replace('.', '_'))
        if not os.path.exists(out_h5_folder):
            os.makedirs(f"{out_h5_folder}/color")
            os.makedirs(f"{out_h5_folder}/depth")
            os.makedirs(f"{out_h5_folder}/ir")
            os.makedirs(f"{out_h5_folder}/thermal")
        h5_path = os.path.join(whole_folder, h5_file)
        extract_hdf5(h5_path, out_h5_folder, 'color')
        # extract_hdf5(h5_path, out_h5_folder, 'depth')
        # extract_hdf5(h5_path, out_h5_folder, 'ir')
        # extract_hdf5(h5_path, out_h5_folder, 'thermal')

