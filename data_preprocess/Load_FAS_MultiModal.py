from __future__ import print_function, division
import os
import torch
import pandas as pd
# from skimage import io, transform
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math
import os
import imgaug.augmenters as iaa

# face_scale = 0.9  #default for test, for training , can be set from [0.8 to 1.0]

# data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
seq = iaa.Sequential([
    iaa.Add(value=(-40, 40), per_channel=True),  # Add color
    iaa.GammaContrast(gamma=(0.5, 1.5))  # GammaContrast with a gamma of 0.5 to 1.5
])


class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """

    def __call__(self, sample):
        image_x, image_x_depth, image_x_ir, image_x_ir_HOG, image_x_ir_PLGF, spoofing_label, map_x1 = sample['image_x'], \
                                                                                                      sample[
                                                                                                          'image_x_depth'], \
                                                                                                      sample[
                                                                                                          'image_x_ir'], \
                                                                                                      sample[
                                                                                                          'image_x_ir_HOG'], \
                                                                                                      sample[
                                                                                                          'image_x_ir_PLGF'], \
                                                                                                      sample[
                                                                                                          'spoofing_label'], \
                                                                                                      sample['map_x1']

        new_image_x = (image_x - 127.5) / 128  # [-1,1]
        new_image_x_depth = (image_x_depth - 127.5) / 128  # [-1,1]
        new_image_x_ir = (image_x_ir - 127.5) / 128  # [-1,1]

        new_image_x_ir_HOG = (image_x_ir_HOG - 127.5) / 128  # [-1,1]
        new_image_x_ir_PLGF = (image_x_ir_PLGF - 127.5) / 128  # [-1,1]

        return {'image_x': new_image_x, 'image_x_depth': new_image_x_depth, 'image_x_ir': new_image_x_ir,
                'image_x_ir_HOG': new_image_x_ir_HOG, 'image_x_ir_PLGF': new_image_x_ir_PLGF,
                'spoofing_label': spoofing_label, 'map_x1': map_x1}


class RandomHorizontalFlip(object):
    """Horizontally flip the given Image randomly with a probability of 0.5."""

    def __call__(self, sample):
        image_x, image_x_depth, image_x_ir, image_x_ir_HOG, image_x_ir_PLGF, spoofing_label, map_x1 = sample['image_x'], \
                                                                                                      sample[
                                                                                                          'image_x_depth'], \
                                                                                                      sample[
                                                                                                          'image_x_ir'], \
                                                                                                      sample[
                                                                                                          'image_x_ir_HOG'], \
                                                                                                      sample[
                                                                                                          'image_x_ir_PLGF'], \
                                                                                                      sample[
                                                                                                          'spoofing_label'], \
                                                                                                      sample['map_x1']

        new_image_x = np.zeros((224, 224, 3))
        new_image_x_depth = np.zeros((224, 224, 3))
        new_image_x_ir = np.zeros((224, 224, 3))

        new_image_x_ir_HOG = np.zeros((224, 224, 3))
        new_image_x_ir_PLGF = np.zeros((224, 224, 3))

        p = random.random()
        if p < 0.5:
            # print('Flip')

            new_image_x = cv2.flip(image_x, 1)
            new_image_x_depth = cv2.flip(image_x_depth, 1)
            new_image_x_ir = cv2.flip(image_x_ir, 1)

            new_image_x_ir_HOG = cv2.flip(image_x_ir_HOG, 1)
            new_image_x_ir_PLGF = cv2.flip(image_x_ir_PLGF, 1)

            return {'image_x': new_image_x, 'image_x_depth': new_image_x_depth, 'image_x_ir': new_image_x_ir,
                    'image_x_ir_HOG': new_image_x_ir_HOG, 'image_x_ir_PLGF': new_image_x_ir_PLGF,
                    'spoofing_label': spoofing_label, 'map_x1': map_x1}
        else:
            # print('no Flip')
            return {'image_x': image_x, 'image_x_depth': image_x_depth, 'image_x_ir': image_x_ir,
                    'image_x_ir_HOG': image_x_ir_HOG, 'image_x_ir_PLGF': image_x_ir_PLGF,
                    'spoofing_label': spoofing_label, 'map_x1': map_x1}


class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        image_x, image_x_depth, image_x_ir, image_x_ir_HOG, image_x_ir_PLGF, spoofing_label, map_x1 = sample['image_x'], \
                                                                                                      sample[
                                                                                                          'image_x_depth'], \
                                                                                                      sample[
                                                                                                          'image_x_ir'], \
                                                                                                      sample[
                                                                                                          'image_x_ir_HOG'], \
                                                                                                      sample[
                                                                                                          'image_x_ir_PLGF'], \
                                                                                                      sample[
                                                                                                          'spoofing_label'], \
                                                                                                      sample['map_x1']

        # swap color axis because
        # numpy image: (batch_size) x H x W x C
        # torch image: (batch_size) x C X H X W
        image_x = image_x[:, :, ::-1].transpose((2, 0, 1))
        image_x = np.array(image_x)

        image_x_depth = image_x_depth[:, :, ::-1].transpose((2, 0, 1))
        image_x_depth = np.array(image_x_depth)

        image_x_ir = image_x_ir[:, :, ::-1].transpose((2, 0, 1))
        image_x_ir = np.array(image_x_ir)

        image_x_ir_HOG = image_x_ir_HOG[:, :, ::-1].transpose((2, 0, 1))
        image_x_ir_HOG = np.array(image_x_ir_HOG)

        image_x_ir_PLGF = image_x_ir_PLGF[:, :, ::-1].transpose((2, 0, 1))
        image_x_ir_PLGF = np.array(image_x_ir_PLGF)

        map_x1 = np.array(map_x1)

        spoofing_label_np = np.array([0], dtype=np.longlong)
        spoofing_label_np[0] = spoofing_label

        return {'image_x': torch.from_numpy(image_x.astype(np.float64)).float(),
                'image_x_depth': torch.from_numpy(image_x_depth.astype(np.float64)).float(),
                'image_x_ir': torch.from_numpy(image_x_ir.astype(np.float64)).float(),
                'image_x_ir_HOG': torch.from_numpy(image_x_ir_HOG.astype(np.float64)).float(),
                'image_x_ir_PLGF': torch.from_numpy(image_x_ir_PLGF.astype(np.float64)).float(),
                'spoofing_label': torch.from_numpy(spoofing_label_np.astype(np.longlong)).long(),
                'map_x1': torch.from_numpy(map_x1.astype(np.float64)).float()}


class Spoofing_train(Dataset):

    def __init__(self, info_list, root_dir, transform=None):
        super(Spoofing_train, self).__init__()
        self.landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        # print(self.landmarks_frame, len(self.landmarks_frame))
        self.root_dir = root_dir
        self.transform = transform
        # print(len(self.landmarks_frame))

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        # print(self.landmarks_frame.iloc[idx, 0])
        videoname = str(self.landmarks_frame.iloc[idx, 0])
        videoname = videoname.replace('WMCA-Image/R_CDIT/', '')

        rgb_path = videoname[:-14] + 'color/' + videoname[-8:]
        depth_path = videoname[:-14] + 'depth/' + videoname[-8:]
        ir_path = videoname[:-14] + 'ir/' + videoname[-8:]

        # print(idx, rgb_path, depth_path)
        # HOG_ir_path = videoname[:-14] + 'HOG_ir/' + videoname[-8:]
        # HOG_ir_path = videoname[:17] + '_HOG224_16' + videoname[17:-14] + 'HOG_ir/' + videoname[-8:]
        # PLGF_ir_path = videoname[:-14] + 'PLGF_ir/' + videoname[-8:]
        HOG_ir_path = videoname[:-14] + 'HOG_ir/' + videoname[-8:]
        PLGF_ir_path = videoname[:-14] + 'PLGF_ir/' + videoname[-8:]

        rgb_path = os.path.join(self.root_dir, rgb_path)
        depth_path = os.path.join(self.root_dir, depth_path)
        ir_path = os.path.join(self.root_dir, ir_path)

        HOG_ir_path = os.path.join(self.root_dir, HOG_ir_path)
        PLGF_ir_path = os.path.join(self.root_dir, PLGF_ir_path)


        image_x, map_x1 = self.get_single_image_x_RGB(rgb_path)
        image_x_depth = self.get_single_image_x(depth_path)
        image_x_ir = self.get_single_image_x(ir_path)

        image_x_ir_HOG = self.get_single_image_x(HOG_ir_path)
        image_x_ir_PLGF = self.get_single_image_x(PLGF_ir_path)

        spoofing_label = self.landmarks_frame.iloc[idx, 3]
        if spoofing_label == 1:  # real
            spoofing_label = 1  # real
            # map_x1 = np.zeros((28, 28))   # real
            # map_x1 = np.ones((28, 28))
        else:  # fake
            spoofing_label = 0
            # map_x1 = np.ones((28, 28))    # fake
            map_x1 = np.zeros((28, 28))

        sample = {'image_x': image_x, 'image_x_depth': image_x_depth, 'image_x_ir': image_x_ir,
                  'image_x_ir_HOG': image_x_ir_HOG, 'image_x_ir_PLGF': image_x_ir_PLGF,
                  'spoofing_label': spoofing_label, 'map_x1': map_x1}
        # sample = {'image_x': image_x, 'image_x_depth': image_x_depth, 'image_x_ir': image_x_ir,
        #           'spoofing_label': spoofing_label, 'map_x1': map_x1}

        if self.transform:
            sample = self.transform(sample)
        sample['domain'] = spoofing_label
        return sample

    def get_single_image_x_RGB(self, image_path):

        image_x = np.zeros((224, 224, 3))
        binary_mask = np.zeros((28, 28))

        # RGB
        image_x_temp = cv2.imread(image_path)

        # cv2.imwrite('temp.jpg', image_x_temp)

        image_x = cv2.resize(image_x_temp, (224, 224))

        # data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
        image_x_aug = seq.augment_image(image_x)

        image_x_temp_gray = cv2.imread(image_path, 0)
        image_x_temp_gray = cv2.resize(image_x_temp_gray, (28, 28))
        for i in range(28):
            for j in range(28):
                if image_x_temp_gray[i, j] > 0:
                    binary_mask[i, j] = 1
                else:
                    binary_mask[i, j] = 0

        return image_x_aug, binary_mask

    def get_single_image_x(self, image_path):
        # print(image_path)
        """
        /home/hdd1/share/public_data/FAS-2023/WMCA/image/516_02_014_1_07/HOG_ir/0024.jpg

        """
        image_x = np.zeros((224, 224, 3))

        # RGB
        image_x_temp = cv2.imread(image_path)

        # cv2.imwrite('temp.jpg', image_x_temp)

        image_x = cv2.resize(image_x_temp, (224, 224))

        # data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
        image_x_aug = seq.augment_image(image_x)

        return image_x_aug


class Spoofing_valtest(Dataset):

    def __init__(self, info_list, root_dir, transform=None):

        self.landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        # print(self.landmarks_frame.iloc[idx, 0])
        videoname = str(self.landmarks_frame.iloc[idx, 0])
        videoname = videoname.replace('WMCA-Image/R_CDIT/', '')

        rgb_path = videoname[:-14] + 'color/' + videoname[-8:]
        depth_path = videoname[:-14] + 'depth/' + videoname[-8:]
        ir_path = videoname[:-14] + 'ir/' + videoname[-8:]

        # print(idx, rgb_path, depth_path)
        # HOG_ir_path = videoname[:-14] + 'HOG_ir/' + videoname[-8:]
        # HOG_ir_path = videoname[:17] + '_HOG224_16' + videoname[17:-14] + 'HOG_ir/' + videoname[-8:]
        # PLGF_ir_path = videoname[:-14] + 'PLGF_ir/' + videoname[-8:]
        HOG_ir_path = videoname[:-14] + 'HOG_ir/' + videoname[-8:]
        PLGF_ir_path = videoname[:-14] + 'PLGF_ir/' + videoname[-8:]

        rgb_path = os.path.join(self.root_dir, rgb_path)
        depth_path = os.path.join(self.root_dir, depth_path)
        ir_path = os.path.join(self.root_dir, ir_path)

        HOG_ir_path = os.path.join(self.root_dir, HOG_ir_path)
        PLGF_ir_path = os.path.join(self.root_dir, PLGF_ir_path)

        image_x, map_x1 = self.get_single_image_x_RGB(rgb_path)
        image_x_depth = self.get_single_image_x(depth_path)
        image_x_ir = self.get_single_image_x(ir_path)

        image_x_ir_HOG = self.get_single_image_x(HOG_ir_path)
        image_x_ir_PLGF = self.get_single_image_x(PLGF_ir_path)

        spoofing_label = self.landmarks_frame.iloc[idx, 3]
        if spoofing_label == 1:  # real
            spoofing_label = 1  # real
            # map_x1 = np.zeros((28, 28))   # real
            # map_x1 = np.ones((28, 28))
        else:  # fake
            spoofing_label = 0
            # map_x1 = np.ones((28, 28))    # fake
            map_x1 = np.zeros((28, 28))

        sample = {'image_x': image_x, 'image_x_depth': image_x_depth, 'image_x_ir': image_x_ir,
                  'image_x_ir_HOG': image_x_ir_HOG, 'image_x_ir_PLGF': image_x_ir_PLGF,
                  'spoofing_label': spoofing_label, 'map_x1': map_x1}
        # sample = {'image_x': image_x, 'image_x_depth': image_x_depth, 'image_x_ir': image_x_ir,
        #           'spoofing_label': spoofing_label, 'map_x1': map_x1}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_single_image_x_RGB(self, image_path):

        image_x = np.zeros((224, 224, 3))
        binary_mask = np.zeros((28, 28))

        # RGB
        image_x_temp = cv2.imread(image_path)

        # cv2.imwrite('temp.jpg', image_x_temp)

        image_x = cv2.resize(image_x_temp, (224, 224))

        image_x_temp_gray = cv2.imread(image_path, 0)
        image_x_temp_gray = cv2.resize(image_x_temp_gray, (28, 28))
        for i in range(28):
            for j in range(28):
                if image_x_temp_gray[i, j] > 0:
                    binary_mask[i, j] = 1
                else:
                    binary_mask[i, j] = 0

        return image_x, binary_mask

    def get_single_image_x(self, image_path):

        image_x = np.zeros((224, 224, 3))

        # RGB
        image_x_temp = cv2.imread(image_path)

        # cv2.imwrite('temp.jpg', image_x_temp)

        image_x = cv2.resize(image_x_temp, (224, 224))

        return image_x

