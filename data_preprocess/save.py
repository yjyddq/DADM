# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 18:00:27 2022

@author: ansin
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog as s_hog
import math
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value


def lbp_calculated_pixel(img, x, y):
    center = img[x][y]
    val_ar = []
    # top_left
    val_ar.append(get_pixel(img, center, x - 1, y - 1))
    # top
    val_ar.append(get_pixel(img, center, x - 1, y))
    # top_right
    val_ar.append(get_pixel(img, center, x - 1, y + 1))
    # right
    val_ar.append(get_pixel(img, center, x, y + 1))
    # bottom_right
    val_ar.append(get_pixel(img, center, x + 1, y + 1))
    # bottom
    val_ar.append(get_pixel(img, center, x + 1, y))
    # bottom_left
    val_ar.append(get_pixel(img, center, x + 1, y - 1))
    # left
    val_ar.append(get_pixel(img, center, x, y - 1))

    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]

    return val


def Normalization255_GRAY(img, max_value=255, min_value=0):
    Max = np.max(img)
    Min = np.min(img)
    img = ((img - Min) / (Max - Min)) * (max_value - min_value) + min_value
    return img


def filter_nn(img, kernel, padding=1):
    img = torch.Tensor(img)
    img = img.unsqueeze(0).unsqueeze(0)
    img = img.float()
    kernel = torch.Tensor(kernel)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    weight = nn.Parameter(data=kernel, requires_grad=False)
    res = F.conv2d(img, weight, padding=padding)
    return res


def produce_x(img_gray):
    filter_x = np.array([
        [np.cos(math.atan2(2, -2)) / (pow(2, 2) + pow(-2, 2)),
         np.cos(math.atan2(2, -1)) / (pow(2, 2) + pow(-1, 2)),
         np.cos(math.atan2(2, 0)) / (pow(2, 2) + pow(0, 2)),
         np.cos(math.atan2(2, 1)) / (pow(2, 2) + pow(1, 2)),
         np.cos(math.atan2(2, 2)) / (pow(2, 2) + pow(2, 2))],

        [np.cos(math.atan2(1, -2)) / (pow(1, 2) + pow(-2, 2)),
         np.cos(math.atan2(1, -1)) / (pow(1, 2) + pow(-1, 2)),
         np.cos(math.atan2(1, 0)) / (pow(1, 2) + pow(0, 2)),
         np.cos(math.atan2(1, 1)) / (pow(1, 2) + pow(1, 2)),
         np.cos(math.atan2(1, 2)) / (pow(1, 2) + pow(2, 2))],

        [np.cos(math.atan2(0, -2)) / (pow(0, 2) + pow(-2, 2)),
         np.cos(math.atan2(0, -1)) / (pow(0, 2) + pow(-1, 2)),
         0,
         np.cos(math.atan2(0, 1)) / (pow(0, 2) + pow(1, 2)),
         np.cos(math.atan2(0, 2)) / (pow(0, 2) + pow(2, 2))],

        [np.cos(math.atan2(-1, -2)) / (pow(-1, 2) + pow(-2, 2)),
         np.cos(math.atan2(-1, -1)) / (pow(-1, 2) + pow(-1, 2)),
         np.cos(math.atan2(-1, 0)) / (pow(-1, 2) + pow(0, 2)),
         np.cos(math.atan2(-1, 1)) / (pow(-1, 2) + pow(1, 2)),
         np.cos(math.atan2(-1, 2)) / (pow(-1, 2) + pow(2, 2))],

        [np.cos(math.atan2(-2, -2)) / (pow(-2, 2) + pow(-2, 2)),
         np.cos(math.atan2(-2, -1)) / (pow(-2, 2) + pow(-1, 2)),
         np.cos(math.atan2(-2, 0)) / (pow(-2, 2) + pow(0, 2)),
         np.cos(math.atan2(-2, 1)) / (pow(-2, 2) + pow(1, 2)),
         np.cos(math.atan2(-2, 2)) / (pow(-2, 2) + pow(2, 2))]
    ])

    img_x = filter_nn(img_gray, filter_x, padding=2)
    img_xorl = np.array(img_x).reshape(img_x.shape[2], -1)

    return img_xorl


def produce_y(img_gray):
    filter_y = np.array([
        [np.sin(math.atan2(2, -2)) / (pow(2, 2) + pow(-2, 2)),
         np.sin(math.atan2(2, -1)) / (pow(2, 2) + pow(-1, 2)),
         np.sin(math.atan2(2, 0)) / (pow(2, 2) + pow(0, 2)),
         np.sin(math.atan2(2, 1)) / (pow(2, 2) + pow(1, 2)),
         np.sin(math.atan2(2, 2)) / (pow(2, 2) + pow(2, 2))],

        [np.sin(math.atan2(1, -2)) / (pow(1, 2) + pow(-2, 2)),
         np.sin(math.atan2(1, -1)) / (pow(1, 2) + pow(-1, 2)),
         np.sin(math.atan2(1, 0)) / (pow(1, 2) + pow(0, 2)),
         np.sin(math.atan2(1, 1)) / (pow(1, 2) + pow(1, 2)),
         np.sin(math.atan2(1, 2)) / (pow(1, 2) + pow(2, 2))],

        [np.sin(math.atan2(0, -2)) / (pow(0, 2) + pow(-2, 2)),
         np.sin(math.atan2(0, -1)) / (pow(0, 2) + pow(-1, 2)),
         0,
         np.sin(math.atan2(0, 1)) / (pow(0, 2) + pow(1, 2)),
         np.sin(math.atan2(0, 2)) / (pow(0, 2) + pow(2, 2))],

        [np.sin(math.atan2(-1, -2)) / (pow(-1, 2) + pow(-2, 2)),
         np.sin(math.atan2(-1, -1)) / (pow(-1, 2) + pow(-1, 2)),
         np.sin(math.atan2(-1, 0)) / (pow(-1, 2) + pow(0, 2)),
         np.sin(math.atan2(-1, 1)) / (pow(-1, 2) + pow(1, 2)),
         np.sin(math.atan2(-1, 2)) / (pow(-1, 2) + pow(2, 2))],

        [np.sin(math.atan2(-2, -2)) / (pow(-2, 2) + pow(-2, 2)),
         np.sin(math.atan2(-2, -1)) / (pow(-2, 2) + pow(-1, 2)),
         np.sin(math.atan2(-2, 0)) / (pow(-2, 2) + pow(0, 2)),
         np.sin(math.atan2(-2, 1)) / (pow(-2, 2) + pow(1, 2)),
         np.sin(math.atan2(-2, 2)) / (pow(-2, 2) + pow(2, 2))]
    ])

    img_y = filter_nn(img_gray, filter_y, padding=2)
    img_yorl = np.array(img_y).reshape(img_y.shape[2], -1)

    return img_yorl


def filter_16_2(img):
    img = np.where(img > 2, img, 2)
    img_xorl = produce_x(img)
    img_yorl = produce_y(img)
    magtitude = np.arctan(np.sqrt((np.divide(img_xorl, img + 0.0001) ** 2) + (np.divide(img_yorl, img + 0.0001) ** 2)))
    magtitude = Normalization255_GRAY(magtitude, 255, 1)

    return magtitude


def plgf(I):
    I = filter_16_2(I)
    O = I
    return O
