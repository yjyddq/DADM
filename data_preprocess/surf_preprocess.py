import os
import cv2
import numpy as np
from PIL import Image
from skimage.feature import hog as s_hog
import tqdm
from data_preprocess.save import lbp_calculated_pixel, plgf
import pandas as pd


image_dir = '/mnt/f/SURF/'

txt_list = ['/mnt/f/SURF/yzt_protocol/CASIA-SURF_train.txt',
            '/mnt/f/SURF/yzt_protocol/CASIA-SURF_test.txt',
            '/mnt/f/SURF/yzt_protocol/CASIA-SURF_val.txt']
for txt in txt_list:
    landmarks_frame = pd.read_csv(txt, delimiter=' ', header=None)
    for idx in tqdm.trange(len(landmarks_frame)):
        ir_dir = str(landmarks_frame.iloc[idx, 2]).replace('CASIA-SURF-CROP/', image_dir)
        hog_dir = ir_dir.split('/ir/')[0] + '/HOG_ir'
        plgf_dir = ir_dir.split('/ir/')[0] + '/PLGF_ir'
        if not os.path.exists(hog_dir):
            os.mkdir(hog_dir)
        if not os.path.exists(plgf_dir):
            os.mkdir(plgf_dir)
        # color = str(landmarks_frame.iloc[idx, 0]).replace('CASIA-SURF-CROP/', image_dir)
        # color = str(landmarks_frame.iloc[idx, 0]).replace('CASIA-SURF-CROP/', image_dir)
        example = cv2.imread(ir_dir)
        src_img = Image.open(ir_dir)
        height, width, _ = example.shape
        example_gray = cv2.cvtColor(example, cv2.COLOR_BGR2GRAY)
        example_lbp = np.zeros((height, width), np.uint8)
        for i in range(0, height):
            for j in range(0, width):
                example_lbp[i, j] = lbp_calculated_pixel(example_gray, i, j)
        fd, example_hog = s_hog(example_gray, orientations=9, pixels_per_cell=(8, 8),
                                cells_per_block=(2, 2), visualize=True)
        hog = Image.fromarray(example_hog).convert('RGB')
        hog.save(f"{hog_dir}/{ir_dir.split('/')[-1]}")
        print(hog_dir)
        src_img = src_img.convert('L')
        src_img = np.asarray(src_img)
        tgt_img = plgf(src_img)
        tgt_img = Image.fromarray(tgt_img).convert('RGB')
        tgt_img.save(f"{plgf_dir}/{ir_dir.split('/')[-1]}")


