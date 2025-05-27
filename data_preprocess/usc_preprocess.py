import os
import cv2
import numpy as np
from PIL import Image
from skimage.feature import hog as s_hog
import tqdm
from data_preprocess.save import lbp_calculated_pixel, plgf



npz_dir = '/mnt/f/USC_MULTI_SPECTRAL/Data'
image_dir = '/mnt/f/USC_MULTI_SPECTRAL/image'

for npz_name in tqdm.tqdm(os.listdir(npz_dir)):
    whole_npz_name = os.path.join(npz_dir, npz_name)
    x = np.load(whole_npz_name)['data'][0]
    # print(x.shape)
    """读取rgb, IR, Depth"""
    color = np.ceil(x[0:3] * 255)
    ir = np.ceil(x[3:4] * 255)
    depth = np.ceil(x[4:5] * 255)
    id_dir = f"{image_dir}/{npz_name.split('.')[0]}"
    if not os.path.exists(id_dir):
        os.mkdir(id_dir)
    color_dir = f"{id_dir}/color.jpg"
    depth_dir = f"{id_dir}/depth.jpg"
    ir_dir = f"{id_dir}/ir.jpg"
    # cv2.imwrite(color_dir, color.transpose(1, 2, 0))
    # cv2.imwrite(depth_dir, depth.transpose(1, 2, 0))
    # cv2.imwrite(ir_dir, ir.transpose(1, 2, 0))

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
    hog.save(f"{id_dir}/HOG_ir.jpg")
    src_img = src_img.convert('L')
    src_img = np.asarray(src_img)
    tgt_img = plgf(src_img)
    tgt_img = Image.fromarray(tgt_img).convert('RGB')
    tgt_img.save(f"{id_dir}/PLGF_ir.jpg")