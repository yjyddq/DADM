import os
import cv2
import numpy as np
from PIL import Image
from skimage.feature import hog as s_hog
import tqdm
from data_preprocess.save import lbp_calculated_pixel, plgf
import pandas as pd


image_dir = '/mnt/f/'

txt_list = ['/mnt/f/CeFA/yzt/CeFA_train.txt',
            '/mnt/f/CeFA/yzt/CeFA_test.txt',
            '/mnt/f/CeFA/yzt/CeFA_val.txt']

"""去除黑边"""
class RemoveBlackBorders(object):
    def __call__(self, im):
        if type(im) == list:
            return [self.__call__(ims) for ims in im]
        V = np.array(im)
        V = np.mean(V, axis=2)
        X = np.sum(V, axis=0)
        Y = np.sum(V, axis=1)
        y1 = np.nonzero(Y)[0][0]
        y2 = np.nonzero(Y)[0][-1]

        x1 = np.nonzero(X)[0][0]
        x2 = np.nonzero(X)[0][-1]
        return im.crop([x1, y1, x2, y2])

    def __repr__(self):
        return self.__class__.__name__

border_removal = RemoveBlackBorders()

for txt in txt_list:
    landmarks_frame = pd.read_csv(txt, delimiter=' ', header=None)
    for idx in tqdm.trange(len(landmarks_frame)):
        rgb_dir = image_dir + str(landmarks_frame.iloc[idx, 0])
        depth_dir = image_dir + str(landmarks_frame.iloc[idx, 1])
        ir_dir = image_dir + str(landmarks_frame.iloc[idx, 2])
        nb_rgb_dir = rgb_dir.split('/profile/')[0] + '/profile_nb'
        nb_depth_dir = depth_dir.split('/depth/')[0] + '/depth_nb'
        nb_ir_dir = ir_dir.split('/ir/')[0] + '/ir_nb'
        if not os.path.exists(nb_rgb_dir):
            os.mkdir(nb_rgb_dir)
        if not os.path.exists(nb_depth_dir):
            os.mkdir(nb_depth_dir)
        if not os.path.exists(nb_ir_dir):
            os.mkdir(nb_ir_dir)

        example_rgb = Image.open(rgb_dir)
        example_depth = Image.open(depth_dir)
        example_ir = Image.open(ir_dir)

        nb_example_rgb = border_removal(example_rgb)
        nb_example_depth = border_removal(example_depth)
        nb_example_ir = border_removal(example_ir)

        nb_example_rgb.save(f"{nb_rgb_dir}/{rgb_dir.split('/')[-1]}")
        nb_example_depth.save(f"{nb_depth_dir}/{depth_dir.split('/')[-1]}")
        nb_example_ir.save(f"{nb_ir_dir}/{ir_dir.split('/')[-1]}")
