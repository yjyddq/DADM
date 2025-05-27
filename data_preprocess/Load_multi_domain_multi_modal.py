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
                                                                                                      sample['image_x_depth'], \
                                                                                                      sample['image_x_ir'], \
                                                                                                      sample['image_x_ir_HOG'], \
                                                                                                      sample['image_x_ir_PLGF'], \
                                                                                                      sample['spoofing_label'], \
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


class Multi_Domain_Spoofing_train(Dataset):
    def __init__(self, data_list=['WMCA', 'USC', 'CeFA', 'SURF'], transform=None, get_type='all'):
        super(Multi_Domain_Spoofing_train, self).__init__()
        self.len = 0
        self.get_type = get_type
        self.root_dir_dict = {
            "WMCA": "/mnt/f/FASDataset/WMCA/image/",
            "CeFA": "/mnt/f/FASDataset/",
            "SURF": "/mnt/f/FASDataset/SURF/",
            "USC": "/mnt/f/FASDataset/USC_MULTI_SPECTRAL/image/"
        }
        self.info_dict = {
            "WMCA": "/mnt/f/FASDataset/WMCA/yzt/P1/WMCA_Grandtest_train.txt",
            "CeFA": "/mnt/f/FASDataset/CeFA/yzt/CeFA_train.txt",
            "SURF": "/mnt/f/FASDataset/SURF/yzt_protocol/CASIA-SURF_train.txt",
            "USC": "/mnt/f/FASDataset/USC_MULTI_SPECTRAL/USC_MutiModal_Grandtest_Protocol/train.txt"
        }
        self.transform = transform
        self.dir_list = []

        if 'WMCA' in data_list:
            self.load_WMCA(self.info_dict['WMCA'])
        if 'SURF' in data_list:
            self.load_SURF(self.info_dict['SURF'])
        if 'CeFA' in data_list:
            self.load_CeFA(self.info_dict['CeFA'])
        if 'USC' in data_list:
            self.load_USC(self.info_dict['USC'])

        self.domain_id = {
            "True": 1,
            "WMCA": 0,
            "CeFA": 2,
            "SURF": 3,
            "USC": 4,
        }
        self.modality_domain_id = {item: idx for idx, item in enumerate(data_list)}

    def __len__(self):
        return len(self.dir_list)

    def load_WMCA(self, info_list):
        landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        for idx in range(len(landmarks_frame)):
            video_name = str(landmarks_frame.iloc[idx, 0])
            video_name = video_name.replace('WMCA-Image/R_CDIT/', '')

            rgb_path = video_name[:-14] + 'color/' + video_name[-8:]
            depth_path = video_name[:-14] + 'depth/' + video_name[-8:]
            ir_path = video_name[:-14] + 'ir/' + video_name[-8:]

            HOG_ir_path = video_name[:-14] + 'HOG_ir/' + video_name[-8:]
            PLGF_ir_path = video_name[:-14] + 'PLGF_ir/' + video_name[-8:]

            rgb_path = os.path.join(self.root_dir_dict['WMCA'], rgb_path)
            depth_path = os.path.join(self.root_dir_dict['WMCA'], depth_path)
            ir_path = os.path.join(self.root_dir_dict['WMCA'], ir_path)

            HOG_ir_path = os.path.join(self.root_dir_dict['WMCA'], HOG_ir_path)
            PLGF_ir_path = os.path.join(self.root_dir_dict['WMCA'], PLGF_ir_path)
            spoofing_label = landmarks_frame.iloc[idx, 3]
            if self.get_type == 'pos' and spoofing_label == 0:
                continue
            if self.get_type == 'neg' and spoofing_label == 1:
                continue
            # spoofing_label: 0 spoof，1 live
            self.dir_list.append({
                "image_x": rgb_path,
                "image_x_depth": depth_path,
                "image_x_ir": ir_path,
                "image_x_ir_HOG": HOG_ir_path,
                "image_x_ir_PLGF": PLGF_ir_path,
                "spoofing_label": spoofing_label,
                "domain": 'WMCA' if spoofing_label == 0 else 'True',
                "modality_domain": 'WMCA'
            })
            # print(rgb_path)

    def load_CeFA(self, info_list):
        landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        for idx in range(len(landmarks_frame)):
            video_name = str(landmarks_frame.iloc[idx, 0])
            video_name = self.root_dir_dict['CeFA'] + video_name

            rgb_path = video_name
            depth_path = video_name.replace('/profile/', '/depth/')
            ir_path = video_name.replace('/profile/', '/ir/')

            HOG_ir_path = video_name.replace('/profile/', '/HOG_ir/')
            PLGF_ir_path = video_name.replace('/profile/', '/PLGF_ir/')

            spoofing_label = landmarks_frame.iloc[idx, 3]
            if self.get_type == 'pos' and spoofing_label == 0:
                continue
            if self.get_type == 'neg' and spoofing_label == 1:
                continue
            # spoofing_label: 0 spoof，1 live
            self.dir_list.append({
                "image_x": rgb_path,
                "image_x_depth": depth_path,
                "image_x_ir": ir_path,
                "image_x_ir_HOG": HOG_ir_path,
                "image_x_ir_PLGF": PLGF_ir_path,
                "spoofing_label": spoofing_label,
                "domain": 'CeFA' if spoofing_label == 0 else 'True',
                "modality_domain": 'CeFA'
            })

    def load_SURF(self, info_list):
        landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        for idx in range(len(landmarks_frame)):
            video_name = str(landmarks_frame.iloc[idx, 0])
            video_name = video_name.replace('CASIA-SURF-CROP/', self.root_dir_dict['SURF'])

            rgb_path = video_name
            depth_path = video_name.replace('/color/', '/depth/')
            ir_path = video_name.replace('/color/', '/ir/')

            HOG_ir_path = video_name.replace('/color/', '/HOG_ir/')
            PLGF_ir_path = video_name.replace('/color/', '/PLGF_ir/')

            spoofing_label = landmarks_frame.iloc[idx, 3]
            if self.get_type == 'pos' and spoofing_label == 0:
                continue
            if self.get_type == 'neg' and spoofing_label == 1:
                continue
            # spoofing_label: 0 spoof，1 live
            self.dir_list.append({
                "image_x": rgb_path,
                "image_x_depth": depth_path,
                "image_x_ir": ir_path,
                "image_x_ir_HOG": HOG_ir_path,
                "image_x_ir_PLGF": PLGF_ir_path,
                "spoofing_label": spoofing_label,
                "domain": 'SURF' if spoofing_label == 0 else 'True',
                "modality_domain": 'SURF'
            })

    def load_USC(self, info_list):
        landmarks_frame = pd.read_csv(info_list, delimiter='	', header=None)
        for idx in range(len(landmarks_frame)):
            video_name = str(landmarks_frame.iloc[idx, 0]).split('.npz')[0]

            rgb_path = video_name + '/color.jpg'
            depth_path = video_name + '/depth.jpg'
            ir_path = video_name + '/ir.jpg'

            HOG_ir_path = video_name + '/HOG_ir.jpg'
            PLGF_ir_path = video_name + '/PLGF_ir.jpg'

            rgb_path = os.path.join(self.root_dir_dict['USC'], rgb_path)
            depth_path = os.path.join(self.root_dir_dict['USC'], depth_path)
            ir_path = os.path.join(self.root_dir_dict['USC'], ir_path)

            HOG_ir_path = os.path.join(self.root_dir_dict['USC'], HOG_ir_path)
            PLGF_ir_path = os.path.join(self.root_dir_dict['USC'], PLGF_ir_path)
            spoofing_label = landmarks_frame.iloc[idx, 1]
            if self.get_type == 'pos' and spoofing_label == 0:
                continue
            if self.get_type == 'neg' and spoofing_label == 1:
                continue
            # spoofing_label: 0 spoof，1 live
            self.dir_list.append({
                "image_x": rgb_path,
                "image_x_depth": depth_path,
                "image_x_ir": ir_path,
                "image_x_ir_HOG": HOG_ir_path,
                "image_x_ir_PLGF": PLGF_ir_path,
                "spoofing_label": spoofing_label,
                "domain": 'USC' if spoofing_label == 0 else 'True',
                "modality_domain": 'USC'
            })

    def __getitem__(self, idx):
        spoofing_label = self.dir_list[idx]['spoofing_label']
        image_x, map_x1 = self.get_single_image_x_RGB(self.dir_list[idx]['image_x'])
        image_x_depth = self.get_single_image_x(self.dir_list[idx]['image_x_depth'])
        image_x_ir = self.get_single_image_x(self.dir_list[idx]['image_x_ir'])
        image_x_ir_HOG = self.get_single_image_x(self.dir_list[idx]['image_x_ir_HOG'])
        image_x_ir_PLGF = self.get_single_image_x(self.dir_list[idx]['image_x_ir_PLGF'])

        if spoofing_label == 1:  # real
            spoofing_label = 1  # real
        else:  # fake
            spoofing_label = 0
            map_x1 = np.zeros((28, 28))

        sample = {'image_x': image_x, 'image_x_depth': image_x_depth, 'image_x_ir': image_x_ir,
                  'image_x_ir_HOG': image_x_ir_HOG, 'image_x_ir_PLGF': image_x_ir_PLGF, 'map_x1': map_x1,
                  'spoofing_label': spoofing_label}
        try:
            if self.transform:
                sample = self.transform(sample)
        except:
            print("Image loading error!!!: ", self.dir_list[idx])
            if self.transform:
                sample = self.transform(sample)
        sample['domain'] = self.domain_id[self.dir_list[idx]['domain']]
        sample['modality_domain'] = self.modality_domain_id[self.dir_list[idx]['modality_domain']]
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

class Multi_Domain_Spoofing_valtest(Dataset):
    def __init__(self, data_list=['WMCA', 'USC', 'CeFA', 'SURF'], transform=None, get_type='all'):
        super(Multi_Domain_Spoofing_valtest, self).__init__()
        self.len = 0
        self.get_type = get_type
        # print(self.landmarks_frame, len(self.landmarks_frame))
        self.root_dir_dict = {
            "WMCA": "/mnt/f/FASDataset/WMCA/image/",
            "CeFA": "/mnt/f/FASDataset/",
            "SURF": "/mnt/f/FASDataset/SURF/",
            "USC": "/mnt/f/FASDataset/USC_MULTI_SPECTRAL/image/"
        }
        self.info_dict = {
            "WMCA": "/mnt/f/FASDataset/WMCA/yzt/P1/WMCA_Grandtest_test.txt",
            "CeFA": "/mnt/f/FASDataset/CeFA/yzt/CeFA_test.txt",
            "SURF": "/mnt/f/FASDataset/SURF/yzt_protocol/CASIA-SURF_test.txt",
            "USC": "/mnt/f/FASDataset/USC_MULTI_SPECTRAL/USC_MutiModal_Grandtest_Protocol/test.txt"
        }
        self.transform = transform
        # print(len(self.landmarks_frame))
        self.dir_list = []
        self.domain_id = {
            "True": 0,
            "WMCA": 1,
            "CeFA": 2,
            "SURF": 3,
            "USC": 4
        }
        self.modality_domain_id = {item: idx for idx, item in enumerate(data_list)}
        if 'WMCA' in data_list:
            self.load_WMCA(self.info_dict['WMCA'])
        if 'SURF' in data_list:
            self.load_SURF(self.info_dict['SURF'])
        if 'CeFA' in data_list:
            self.load_CeFA(self.info_dict['CeFA'])
        if 'USC' in data_list:
            self.load_USC(self.info_dict['USC'])

    def __len__(self):
        return len(self.dir_list)

    def load_WMCA(self, info_list):
        landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        for idx in range(len(landmarks_frame)):
            video_name = str(landmarks_frame.iloc[idx, 0])
            video_name = video_name.replace('WMCA-Image/R_CDIT/', '')

            rgb_path = video_name[:-14] + 'color/' + video_name[-8:]
            depth_path = video_name[:-14] + 'depth/' + video_name[-8:]
            ir_path = video_name[:-14] + 'ir/' + video_name[-8:]

            HOG_ir_path = video_name[:-14] + 'HOG_ir/' + video_name[-8:]
            PLGF_ir_path = video_name[:-14] + 'PLGF_ir/' + video_name[-8:]

            rgb_path = os.path.join(self.root_dir_dict['WMCA'], rgb_path)
            depth_path = os.path.join(self.root_dir_dict['WMCA'], depth_path)
            ir_path = os.path.join(self.root_dir_dict['WMCA'], ir_path)

            HOG_ir_path = os.path.join(self.root_dir_dict['WMCA'], HOG_ir_path)
            PLGF_ir_path = os.path.join(self.root_dir_dict['WMCA'], PLGF_ir_path)
            spoofing_label = landmarks_frame.iloc[idx, 3]
            if self.get_type == 'pos' and spoofing_label == 0:
                continue
            if self.get_type == 'neg' and spoofing_label == 1:
                continue
            """WMCA中spoofing_label为0的是假脸，为1的是真脸"""
            self.dir_list.append({
                "image_x": rgb_path,
                "image_x_depth": depth_path,
                "image_x_ir": ir_path,
                "image_x_ir_HOG": HOG_ir_path,
                "image_x_ir_PLGF": PLGF_ir_path,
                "spoofing_label": spoofing_label,
                "domain": 'WMCA' if spoofing_label == 0 else 'True',
                "modality_domain": 'WMCA'
            })
            # print(rgb_path)

    def load_CeFA(self, info_list):
        landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        for idx in range(len(landmarks_frame)):
            video_name = str(landmarks_frame.iloc[idx, 0])
            video_name = self.root_dir_dict['CeFA'] + video_name

            rgb_path = video_name
            depth_path = video_name.replace('/profile/', '/depth/')
            ir_path = video_name.replace('/profile/', '/ir/')

            # # 去除黑边版本
            # rgb_path = video_name.replace('/profile/', '/profile_nb/')
            # depth_path = video_name.replace('/profile/', '/depth_nb/')
            # ir_path = video_name.replace('/profile/', '/ir_nb/')

            HOG_ir_path = video_name.replace('/profile/', '/HOG_ir/')
            PLGF_ir_path = video_name.replace('/profile/', '/PLGF_ir/')

            spoofing_label = landmarks_frame.iloc[idx, 3]
            if self.get_type == 'pos' and spoofing_label == 0:
                continue
            if self.get_type == 'neg' and spoofing_label == 1:
                continue
            self.dir_list.append({
                "image_x": rgb_path,
                "image_x_depth": depth_path,
                "image_x_ir": ir_path,
                "image_x_ir_HOG": HOG_ir_path,
                "image_x_ir_PLGF": PLGF_ir_path,
                "spoofing_label": spoofing_label,
                "domain": 'CeFA' if spoofing_label == 0 else 'True',
                "modality_domain": 'CeFA'
            })

    def load_SURF(self, info_list):
        landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        for idx in range(len(landmarks_frame)):
            video_name = str(landmarks_frame.iloc[idx, 0])
            video_name = video_name.replace('CASIA-SURF-CROP/', self.root_dir_dict['SURF'])

            rgb_path = video_name
            depth_path = video_name.replace('/color/', '/depth/')
            ir_path = video_name.replace('/color/', '/ir/')

            HOG_ir_path = video_name.replace('/color/', '/HOG_ir/')
            PLGF_ir_path = video_name.replace('/color/', '/PLGF_ir/')

            spoofing_label = landmarks_frame.iloc[idx, 3]
            if self.get_type == 'pos' and spoofing_label == 0:
                continue
            if self.get_type == 'neg' and spoofing_label == 1:
                continue
            self.dir_list.append({
                "image_x": rgb_path,
                "image_x_depth": depth_path,
                "image_x_ir": ir_path,
                "image_x_ir_HOG": HOG_ir_path,
                "image_x_ir_PLGF": PLGF_ir_path,
                "spoofing_label": spoofing_label,
                "domain": 'SURF' if spoofing_label == 0 else 'True',
                "modality_domain": 'SURF'
            })

    def load_USC(self, info_list):
        landmarks_frame = pd.read_csv(info_list, delimiter='	', header=None)
        for idx in range(len(landmarks_frame)):
            video_name = str(landmarks_frame.iloc[idx, 0]).split('.npz')[0]

            rgb_path = video_name + '/color.jpg'
            depth_path = video_name + '/depth.jpg'
            ir_path = video_name + '/ir.jpg'

            HOG_ir_path = video_name + '/HOG_ir.jpg'
            PLGF_ir_path = video_name + '/PLGF_ir.jpg'

            rgb_path = os.path.join(self.root_dir_dict['USC'], rgb_path)
            depth_path = os.path.join(self.root_dir_dict['USC'], depth_path)
            ir_path = os.path.join(self.root_dir_dict['USC'], ir_path)

            HOG_ir_path = os.path.join(self.root_dir_dict['USC'], HOG_ir_path)
            PLGF_ir_path = os.path.join(self.root_dir_dict['USC'], PLGF_ir_path)
            spoofing_label = landmarks_frame.iloc[idx, 1]
            if self.get_type == 'pos' and spoofing_label == 0:
                continue
            if self.get_type == 'neg' and spoofing_label == 1:
                continue
            self.dir_list.append({
                "image_x": rgb_path,
                "image_x_depth": depth_path,
                "image_x_ir": ir_path,
                "image_x_ir_HOG": HOG_ir_path,
                "image_x_ir_PLGF": PLGF_ir_path,
                "spoofing_label": spoofing_label,
                "domain": 'USC' if spoofing_label == 0 else 'True',
                "modality_domain": 'USC'
            })

    def __getitem__(self, idx):
        spoofing_label = self.dir_list[idx]['spoofing_label']
        image_x, map_x1 = self.get_single_image_x_RGB(self.dir_list[idx]['image_x'])
        image_x_depth = self.get_single_image_x(self.dir_list[idx]['image_x_depth'])
        image_x_ir = self.get_single_image_x(self.dir_list[idx]['image_x_ir'])
        image_x_ir_HOG = self.get_single_image_x(self.dir_list[idx]['image_x_ir_HOG'])
        image_x_ir_PLGF = self.get_single_image_x(self.dir_list[idx]['image_x_ir_PLGF'])

        if spoofing_label == 1:  # real
            spoofing_label = 1  # real
        else:  # fake
            spoofing_label = 0
            map_x1 = np.zeros((28, 28))

        sample = {'image_x': image_x, 'image_x_depth': image_x_depth, 'image_x_ir': image_x_ir,
                  'image_x_ir_HOG': image_x_ir_HOG, 'image_x_ir_PLGF': image_x_ir_PLGF, 'map_x1': map_x1,
                  'spoofing_label': spoofing_label, 'domain': self.domain_id[self.dir_list[idx]['domain']],
                  'modality_domain': self.modality_domain_id[self.dir_list[idx]['modality_domain']]}

        if self.transform:
            sample = self.transform(sample)

        sample['domain'] = self.domain_id[self.dir_list[idx]['domain']]
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

class Flexible_Multi_Domain_Spoofing_train(Dataset):
    def __init__(self, data_list=['WMCA', 'USC', 'CeFA', 'SURF'], transform=None, drop_modal_rate=0.3, get_type='all'):
        super(Flexible_Multi_Domain_Spoofing_train, self).__init__()
        # print(data_list)
        self.len = 0
        # print(self.landmarks_frame, len(self.landmarks_frame))
        self.root_dir_dict = {
            "WMCA": "/mnt/f/FASDataset/WMCA/image/",
            "CeFA": "/mnt/f/FASDataset/",
            "SURF": "/mnt/f/FASDataset/SURF/",
            "USC": "/mnt/f/FASDataset/USC_MULTI_SPECTRAL/image/"
        }
        self.info_dict = {
            "WMCA": "/mnt/f/FASDataset/WMCA/yzt/P1/WMCA_Grandtest_train.txt",
            "CeFA": "/mnt/f/FASDataset/CeFA/yzt/CeFA_train.txt",
            "SURF": "/mnt/f/FASDataset/SURF/yzt_protocol/CASIA-SURF_train.txt",
            "USC": "/mnt/f/FASDataset/USC_MULTI_SPECTRAL/USC_MutiModal_Grandtest_Protocol/train.txt"
        }
        self.transform = transform
        # print(len(self.landmarks_frame))
        self.dir_list = []
        self.domain_id = {
            "True": 1,
            "WMCA": 0,
            "CeFA": 2,
            "SURF": 3,
            "USC": 4,
        }
        self.modality_domain_id = {item: idx for idx, item in enumerate(data_list)}
        self.get_type = get_type
        self.drop_modal_rate = drop_modal_rate
        if 'WMCA' in data_list:
            self.load_WMCA(self.info_dict['WMCA'])
        if 'WMCA_no_D' in data_list:
            self.load_WMCA(self.info_dict['WMCA'], missing_modal=['D'])
        if 'WMCA_no_IR' in data_list:
            self.load_WMCA(self.info_dict['WMCA'], missing_modal=['IR'])
        if 'WMCA_no_D_IR' in data_list:
            self.load_WMCA(self.info_dict['WMCA'], missing_modal=['D', 'IR'])

        if 'SURF' in data_list:
            self.load_SURF(self.info_dict['SURF'])
        if 'SURF_no_D' in data_list:
            self.load_SURF(self.info_dict['SURF'], missing_modal=['D'])
        if 'SURF_no_IR' in data_list:
            self.load_SURF(self.info_dict['SURF'], missing_modal=['IR'])
        if 'SURF_no_D_IR' in data_list:
            self.load_SURF(self.info_dict['SURF'], missing_modal=['D', 'IR'])

        if 'CeFA' in data_list:
            self.load_CeFA(self.info_dict['CeFA'])
        if 'CeFA_no_D' in data_list:
            self.load_CeFA(self.info_dict['CeFA'], missing_modal=['D'])
        if 'CeFA_no_IR' in data_list:
            self.load_CeFA(self.info_dict['CeFA'], missing_modal=['IR'])
        if 'CeFA_no_D_IR' in data_list:
            self.load_CeFA(self.info_dict['CeFA'], missing_modal=['D', 'IR'])

        if 'USC' in data_list:
            self.load_USC(self.info_dict['USC'])
        if 'USC_no_D' in data_list:
            self.load_USC(self.info_dict['USC'], missing_modal=['D'])
        if 'USC_no_IR' in data_list:
            self.load_USC(self.info_dict['USC'], missing_modal=['IR'])
        if 'USC_no_D_IR' in data_list:
            self.load_USC(self.info_dict['USC'], missing_modal=['D', 'IR'])

    def __len__(self):
        return len(self.dir_list)

    def load_WMCA(self, info_list, missing_modal=None):
        landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        for idx in range(len(landmarks_frame)):
            video_name = str(landmarks_frame.iloc[idx, 0])
            video_name = video_name.replace('WMCA-Image/R_CDIT/', '')

            rgb_path = video_name[:-14] + 'color/' + video_name[-8:]
            depth_path = video_name[:-14] + 'depth/' + video_name[-8:]
            ir_path = video_name[:-14] + 'ir/' + video_name[-8:]

            HOG_ir_path = video_name[:-14] + 'HOG_ir/' + video_name[-8:]
            PLGF_ir_path = video_name[:-14] + 'PLGF_ir/' + video_name[-8:]

            rgb_path = os.path.join(self.root_dir_dict['WMCA'], rgb_path)
            depth_path = os.path.join(self.root_dir_dict['WMCA'], depth_path)
            ir_path = os.path.join(self.root_dir_dict['WMCA'], ir_path)

            HOG_ir_path = os.path.join(self.root_dir_dict['WMCA'], HOG_ir_path)
            PLGF_ir_path = os.path.join(self.root_dir_dict['WMCA'], PLGF_ir_path)
            spoofing_label = landmarks_frame.iloc[idx, 3]

            if random.uniform(0.0, 1.0) <= self.drop_modal_rate:
                depth_path = None
                ir_path = None
                HOG_ir_path = None
                PLGF_ir_path = None

            if missing_modal is not None:
                if 'D' in missing_modal:
                    if random.uniform(0.0, 1.0) <= self.drop_modal_rate:
                        depth_path = None
                if 'IR' in missing_modal:
                    if random.uniform(0.0, 1.0) <= self.drop_modal_rate:
                        ir_path = None
                        HOG_ir_path = None
                        PLGF_ir_path = None
            if self.get_type == 'pos' and spoofing_label == 0:
                continue
            if self.get_type == 'neg' and spoofing_label == 1:
                continue

            modality_domain = 'WMCA'
            if 'D' in missing_modal and 'IR' not in missing_modal:
                modality_domain = 'WMCA_no_D'
            if 'IR' in missing_modal and 'D' not in missing_modal:
                modality_domain = 'WMCA_no_IR'
            if 'D' in missing_modal and 'IR' in missing_modal:
                modality_domain = 'WMCA_no_D_IR'
            self.dir_list.append({
                "image_x": rgb_path,
                "image_x_depth": depth_path,
                "image_x_ir": ir_path,
                "image_x_ir_HOG": HOG_ir_path,
                "image_x_ir_PLGF": PLGF_ir_path,
                "spoofing_label": spoofing_label,
                "domain": 'WMCA' if spoofing_label == 0 else 'True',
                "modality_domain": modality_domain
            })
            # print(rgb_path)

    def load_CeFA(self, info_list, missing_modal=None):
        landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        for idx in range(len(landmarks_frame)):
            video_name = str(landmarks_frame.iloc[idx, 0])
            video_name = self.root_dir_dict['CeFA'] + video_name

            rgb_path = video_name
            depth_path = video_name.replace('/color/', '/depth/')
            ir_path = video_name.replace('/color/', '/ir/')

            HOG_ir_path = video_name.replace('/color/', '/HOG_ir/')
            PLGF_ir_path = video_name.replace('/color/', '/PLGF_ir/')

            if random.uniform(0.0, 1.0) <= self.drop_modal_rate:
                depth_path = None
                ir_path = None
                HOG_ir_path = None
                PLGF_ir_path = None

            if missing_modal is not None:
                if 'D' in missing_modal:
                    if random.uniform(0.0, 1.0) <= self.drop_modal_rate:
                        depth_path = None
                if 'IR' in missing_modal:
                    if random.uniform(0.0, 1.0) <= self.drop_modal_rate:
                        ir_path = None
                        HOG_ir_path = None
                        PLGF_ir_path = None
            spoofing_label = landmarks_frame.iloc[idx, 3]
            if self.get_type == 'pos' and spoofing_label == 0:
                continue
            if self.get_type == 'neg' and spoofing_label == 1:
                continue

            modality_domain = 'CeFA'
            if 'D' in missing_modal and 'IR' not in missing_modal:
                modality_domain = 'CeFA_no_D'
            if 'IR' in missing_modal and 'D' not in missing_modal:
                modality_domain = 'CeFA_no_IR'
            if 'D' in missing_modal and 'IR' in missing_modal:
                modality_domain = 'CeFA_no_D_IR'
            self.dir_list.append({
                "image_x": rgb_path,
                "image_x_depth": depth_path,
                "image_x_ir": ir_path,
                "image_x_ir_HOG": HOG_ir_path,
                "image_x_ir_PLGF": PLGF_ir_path,
                "spoofing_label": spoofing_label,
                "domain": 'CeFA' if spoofing_label == 0 else 'True',
                "modality_domain": modality_domain
            })

    def load_SURF(self, info_list, missing_modal=None):
        landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        for idx in range(len(landmarks_frame)):
            video_name = str(landmarks_frame.iloc[idx, 0])
            video_name = video_name.replace('CASIA-SURF-CROP/', self.root_dir_dict['SURF'])

            rgb_path = video_name
            depth_path = video_name.replace('/color/', '/depth/')
            ir_path = video_name.replace('/color/', '/ir/')

            HOG_ir_path = video_name.replace('/color/', '/HOG_ir/')
            PLGF_ir_path = video_name.replace('/color/', '/PLGF_ir/')

            spoofing_label = landmarks_frame.iloc[idx, 3]

            if random.uniform(0.0, 1.0) <= self.drop_modal_rate:
                depth_path = None
                ir_path = None
                HOG_ir_path = None
                PLGF_ir_path = None

            if missing_modal is not None:
                if 'D' in missing_modal:
                    if random.uniform(0.0, 1.0) <= self.drop_modal_rate:
                        depth_path = None
                if 'IR' in missing_modal:
                    if random.uniform(0.0, 1.0) <= self.drop_modal_rate:
                        ir_path = None
                        HOG_ir_path = None
                        PLGF_ir_path = None
            if self.get_type == 'pos' and spoofing_label == 0:
                continue
            if self.get_type == 'neg' and spoofing_label == 1:
                continue

            modality_domain = 'SURF'
            if 'D' in missing_modal and 'IR' not in missing_modal:
                modality_domain = 'SURF_no_D'
            if 'IR' in missing_modal and 'D' not in missing_modal:
                modality_domain = 'SURF_no_IR'
            if 'D' in missing_modal and 'IR' in missing_modal:
                modality_domain = 'SURF_no_D_IR'
            self.dir_list.append({
                "image_x": rgb_path,
                "image_x_depth": depth_path,
                "image_x_ir": ir_path,
                "image_x_ir_HOG": HOG_ir_path,
                "image_x_ir_PLGF": PLGF_ir_path,
                "spoofing_label": spoofing_label,
                "domain": 'SURF' if spoofing_label == 0 else 'True',
                "modality_domain": modality_domain
            })

    def load_USC(self, info_list, missing_modal=None):
        landmarks_frame = pd.read_csv(info_list, delimiter='	', header=None)
        for idx in range(len(landmarks_frame)):
            video_name = str(landmarks_frame.iloc[idx, 0]).split('.npz')[0]

            rgb_path = video_name + '/color.jpg'
            depth_path = video_name + '/depth.jpg'
            ir_path = video_name + '/ir.jpg'

            HOG_ir_path = video_name + '/HOG_ir.jpg'
            PLGF_ir_path = video_name + '/PLGF_ir.jpg'

            rgb_path = os.path.join(self.root_dir_dict['USC'], rgb_path)
            depth_path = os.path.join(self.root_dir_dict['USC'], depth_path)
            ir_path = os.path.join(self.root_dir_dict['USC'], ir_path)

            HOG_ir_path = os.path.join(self.root_dir_dict['USC'], HOG_ir_path)
            PLGF_ir_path = os.path.join(self.root_dir_dict['USC'], PLGF_ir_path)
            spoofing_label = landmarks_frame.iloc[idx, 1]

            if random.uniform(0.0, 1.0) <= self.drop_modal_rate:
                depth_path = None
                ir_path = None
                HOG_ir_path = None
                PLGF_ir_path = None

            if missing_modal is not None:
                if 'D' in missing_modal:
                    if random.uniform(0.0, 1.0) <= self.drop_modal_rate:
                        depth_path = None
                if 'IR' in missing_modal:
                    if random.uniform(0.0, 1.0) <= self.drop_modal_rate:
                        ir_path = None
                        HOG_ir_path = None
                        PLGF_ir_path = None
            if self.get_type == 'pos' and spoofing_label == 0:
                continue
            if self.get_type == 'neg' and spoofing_label == 1:
                continue

            modality_domain = 'USC'
            if 'D' in missing_modal and 'IR' not in missing_modal:
                modality_domain = 'USC_no_D'
            if 'IR' in missing_modal and 'D' not in missing_modal:
                modality_domain = 'USC_no_IR'
            if 'D' in missing_modal and 'IR' in missing_modal:
                modality_domain = 'USC_no_D_IR'
            self.dir_list.append({
                "image_x": rgb_path,
                "image_x_depth": depth_path,
                "image_x_ir": ir_path,
                "image_x_ir_HOG": HOG_ir_path,
                "image_x_ir_PLGF": PLGF_ir_path,
                "spoofing_label": spoofing_label,
                "domain": 'USC' if spoofing_label == 0 else 'True',
                "modality_domain": modality_domain
            })

    def __getitem__(self, idx):
        spoofing_label = self.dir_list[idx]['spoofing_label']
        image_x, map_x1 = self.get_single_image_x_RGB(self.dir_list[idx]['image_x'])
        image_x_depth = self.get_single_image_x(self.dir_list[idx]['image_x_depth'])
        image_x_ir = self.get_single_image_x(self.dir_list[idx]['image_x_ir'])
        image_x_ir_HOG = self.get_single_image_x(self.dir_list[idx]['image_x_ir_HOG'])
        image_x_ir_PLGF = self.get_single_image_x(self.dir_list[idx]['image_x_ir_PLGF'])

        if spoofing_label == 1:  # real
            spoofing_label = 1  # real
        else:  # fake
            spoofing_label = 0
            map_x1 = np.zeros((28, 28))

        sample = {'image_x': image_x, 'image_x_depth': image_x_depth, 'image_x_ir': image_x_ir,
                  'image_x_ir_HOG': image_x_ir_HOG, 'image_x_ir_PLGF': image_x_ir_PLGF, 'map_x1': map_x1,
                  'spoofing_label': spoofing_label}
        try:
            if self.transform:
                sample = self.transform(sample)
        except:
            print(self.dir_list[idx])
            if self.transform:
                sample = self.transform(sample)
        sample['domain'] = self.domain_id[self.dir_list[idx]['domain']]
        sample['modality_domain'] = self.modality_domain_id[self.dir_list[idx]['modality_domain']]
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
        if image_path is None:
            image_x = np.zeros((224, 224, 3))
            return image_x
        else:
            # RGB
            image_x_temp = cv2.imread(image_path)

            # cv2.imwrite('temp.jpg', image_x_temp)

            image_x = cv2.resize(image_x_temp, (224, 224))

            # data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
            image_x_aug = seq.augment_image(image_x)

        return image_x_aug

class Flexible_Multi_Domain_Spoofing_valtest(Dataset):
    def __init__(self, data_list=['WMCA', 'USC', 'CeFA', 'SURF'], transform=None, drop_modal_rate=0.7, get_type='all'):
        super(Flexible_Multi_Domain_Spoofing_valtest, self).__init__()
        # print(data_list)
        self.len = 0
        # print(self.landmarks_frame, len(self.landmarks_frame))
        self.root_dir_dict = {
            "WMCA": "/mnt/f/FASDataset/WMCA/image/",
            "CeFA": "/mnt/f/FASDataset/",
            "SURF": "/mnt/f/FASDataset/SURF/",
            "USC": "/mnt/f/FASDataset/USC_MULTI_SPECTRAL/image/"
        }
        self.info_dict = {
            "WMCA": "/mnt/f/FASDataset/WMCA/yzt/P1/WMCA_Grandtest_train.txt",
            "CeFA": "/mnt/f/FASDataset/CeFA/yzt/CeFA_train.txt",
            "SURF": "/mnt/f/FASDataset/SURF/yzt_protocol/CASIA-SURF_train.txt",
            "USC": "/mnt/f/FASDataset/USC_MULTI_SPECTRAL/USC_MutiModal_Grandtest_Protocol/train.txt"
        }
        self.transform = transform
        # print(len(self.landmarks_frame))
        self.dir_list = []
        self.domain_id = {
            "True": 1,
            "WMCA": 0,
            "CeFA": 2,
            "SURF": 3,
            "USC": 4,
        }
        self.modality_domain_id = {item: idx for idx, item in enumerate(data_list)}
        self.get_type = get_type
        self.drop_modal_rate = drop_modal_rate
        if 'WMCA' in data_list:
            self.load_WMCA(self.info_dict['WMCA'])
        if 'WMCA_no_D' in data_list:
            self.load_WMCA(self.info_dict['WMCA'], missing_modal=['D'])
        if 'WMCA_no_IR' in data_list:
            self.load_WMCA(self.info_dict['WMCA'], missing_modal=['IR'])
        if 'WMCA_no_D_IR' in data_list:
            self.load_WMCA(self.info_dict['WMCA'], missing_modal=['D', 'IR'])

        if 'SURF' in data_list:
            self.load_SURF(self.info_dict['SURF'])
        if 'SURF_no_D' in data_list:
            self.load_SURF(self.info_dict['SURF'], missing_modal=['D'])
        if 'SURF_no_IR' in data_list:
            self.load_SURF(self.info_dict['SURF'], missing_modal=['IR'])
        if 'SURF_no_D_IR' in data_list:
            self.load_SURF(self.info_dict['SURF'], missing_modal=['D', 'IR'])

        if 'CeFA' in data_list:
            self.load_CeFA(self.info_dict['CeFA'])
        if 'CeFA_no_D' in data_list:
            self.load_CeFA(self.info_dict['CeFA'], missing_modal=['D'])
        if 'CeFA_no_IR' in data_list:
            self.load_CeFA(self.info_dict['CeFA'], missing_modal=['IR'])
        if 'CeFA_no_D_IR' in data_list:
            self.load_CeFA(self.info_dict['CeFA'], missing_modal=['D', 'IR'])

        if 'USC' in data_list:
            self.load_USC(self.info_dict['USC'])
        if 'USC_no_D' in data_list:
            self.load_USC(self.info_dict['USC'], missing_modal=['D'])
        if 'USC_no_IR' in data_list:
            self.load_USC(self.info_dict['USC'], missing_modal=['IR'])
        if 'USC_no_D_IR' in data_list:
            self.load_USC(self.info_dict['USC'], missing_modal=['D', 'IR'])

    def __len__(self):
        return len(self.dir_list)

    def load_WMCA(self, info_list, missing_modal=None):
        landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        for idx in range(len(landmarks_frame)):
            video_name = str(landmarks_frame.iloc[idx, 0])
            video_name = video_name.replace('WMCA-Image/R_CDIT/', '')

            rgb_path = video_name[:-14] + 'color/' + video_name[-8:]
            depth_path = video_name[:-14] + 'depth/' + video_name[-8:]
            ir_path = video_name[:-14] + 'ir/' + video_name[-8:]

            HOG_ir_path = video_name[:-14] + 'HOG_ir/' + video_name[-8:]
            PLGF_ir_path = video_name[:-14] + 'PLGF_ir/' + video_name[-8:]

            rgb_path = os.path.join(self.root_dir_dict['WMCA'], rgb_path)
            depth_path = os.path.join(self.root_dir_dict['WMCA'], depth_path)
            ir_path = os.path.join(self.root_dir_dict['WMCA'], ir_path)

            HOG_ir_path = os.path.join(self.root_dir_dict['WMCA'], HOG_ir_path)
            PLGF_ir_path = os.path.join(self.root_dir_dict['WMCA'], PLGF_ir_path)
            spoofing_label = landmarks_frame.iloc[idx, 3]

            if random.uniform(0.0, 1.0) <= self.drop_modal_rate:
                depth_path = None
                ir_path = None
                HOG_ir_path = None
                PLGF_ir_path = None

            if missing_modal is not None:
                if 'D' in missing_modal:
                    if random.uniform(0.0, 1.0) <= self.drop_modal_rate:
                        depth_path = None
                if 'IR' in missing_modal:
                    if random.uniform(0.0, 1.0) <= self.drop_modal_rate:
                        ir_path = None
                        HOG_ir_path = None
                        PLGF_ir_path = None
            if self.get_type == 'pos' and spoofing_label == 0:
                continue
            if self.get_type == 'neg' and spoofing_label == 1:
                continue

            modality_domain = 'WMCA'
            if 'D' in missing_modal and 'IR' not in missing_modal:
                modality_domain = 'WMCA_no_D'
            if 'IR' in missing_modal and 'D' not in missing_modal:
                modality_domain = 'WMCA_no_IR'
            if 'D' in missing_modal and 'IR' in missing_modal:
                modality_domain = 'WMCA_no_D_IR'
            self.dir_list.append({
                "image_x": rgb_path,
                "image_x_depth": depth_path,
                "image_x_ir": ir_path,
                "image_x_ir_HOG": HOG_ir_path,
                "image_x_ir_PLGF": PLGF_ir_path,
                "spoofing_label": spoofing_label,
                "domain": 'WMCA' if spoofing_label == 0 else 'True',
                "modality_domain": modality_domain
            })
            # print(rgb_path)

    def load_CeFA(self, info_list, missing_modal=None):
        landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        for idx in range(len(landmarks_frame)):
            video_name = str(landmarks_frame.iloc[idx, 0])
            video_name = self.root_dir_dict['CeFA'] + video_name

            rgb_path = video_name
            depth_path = video_name.replace('/color/', '/depth/')
            ir_path = video_name.replace('/color/', '/ir/')

            HOG_ir_path = video_name.replace('/color/', '/HOG_ir/')
            PLGF_ir_path = video_name.replace('/color/', '/PLGF_ir/')

            if random.uniform(0.0, 1.0) <= self.drop_modal_rate:
                depth_path = None
                ir_path = None
                HOG_ir_path = None
                PLGF_ir_path = None

            if missing_modal is not None:
                if 'D' in missing_modal:
                    if random.uniform(0.0, 1.0) <= self.drop_modal_rate:
                        depth_path = None
                if 'IR' in missing_modal:
                    if random.uniform(0.0, 1.0) <= self.drop_modal_rate:
                        ir_path = None
                        HOG_ir_path = None
                        PLGF_ir_path = None
            spoofing_label = landmarks_frame.iloc[idx, 3]
            if self.get_type == 'pos' and spoofing_label == 0:
                continue
            if self.get_type == 'neg' and spoofing_label == 1:
                continue

            modality_domain = 'CeFA'
            if 'D' in missing_modal and 'IR' not in missing_modal:
                modality_domain = 'CeFA_no_D'
            if 'IR' in missing_modal and 'D' not in missing_modal:
                modality_domain = 'CeFA_no_IR'
            if 'D' in missing_modal and 'IR' in missing_modal:
                modality_domain = 'CeFA_no_D_IR'
            self.dir_list.append({
                "image_x": rgb_path,
                "image_x_depth": depth_path,
                "image_x_ir": ir_path,
                "image_x_ir_HOG": HOG_ir_path,
                "image_x_ir_PLGF": PLGF_ir_path,
                "spoofing_label": spoofing_label,
                "domain": 'CeFA' if spoofing_label == 0 else 'True',
                "modality_domain": modality_domain
            })

    def load_SURF(self, info_list, missing_modal=None):
        landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        for idx in range(len(landmarks_frame)):
            video_name = str(landmarks_frame.iloc[idx, 0])
            video_name = video_name.replace('CASIA-SURF-CROP/', self.root_dir_dict['SURF'])

            rgb_path = video_name
            depth_path = video_name.replace('/color/', '/depth/')
            ir_path = video_name.replace('/color/', '/ir/')

            HOG_ir_path = video_name.replace('/color/', '/HOG_ir/')
            PLGF_ir_path = video_name.replace('/color/', '/PLGF_ir/')

            spoofing_label = landmarks_frame.iloc[idx, 3]

            if random.uniform(0.0, 1.0) <= self.drop_modal_rate:
                depth_path = None
                ir_path = None
                HOG_ir_path = None
                PLGF_ir_path = None

            if missing_modal is not None:
                if 'D' in missing_modal:
                    if random.uniform(0.0, 1.0) <= self.drop_modal_rate:
                        depth_path = None
                if 'IR' in missing_modal:
                    if random.uniform(0.0, 1.0) <= self.drop_modal_rate:
                        ir_path = None
                        HOG_ir_path = None
                        PLGF_ir_path = None
            if self.get_type == 'pos' and spoofing_label == 0:
                continue
            if self.get_type == 'neg' and spoofing_label == 1:
                continue

            modality_domain = 'SURF'
            if 'D' in missing_modal and 'IR' not in missing_modal:
                modality_domain = 'SURF_no_D'
            if 'IR' in missing_modal and 'D' not in missing_modal:
                modality_domain = 'SURF_no_IR'
            if 'D' in missing_modal and 'IR' in missing_modal:
                modality_domain = 'SURF_no_D_IR'
            self.dir_list.append({
                "image_x": rgb_path,
                "image_x_depth": depth_path,
                "image_x_ir": ir_path,
                "image_x_ir_HOG": HOG_ir_path,
                "image_x_ir_PLGF": PLGF_ir_path,
                "spoofing_label": spoofing_label,
                "domain": 'SURF' if spoofing_label == 0 else 'True',
                "modality_domain": modality_domain
            })

    def load_USC(self, info_list, missing_modal=None):
        landmarks_frame = pd.read_csv(info_list, delimiter='	', header=None)
        for idx in range(len(landmarks_frame)):
            video_name = str(landmarks_frame.iloc[idx, 0]).split('.npz')[0]

            rgb_path = video_name + '/color.jpg'
            depth_path = video_name + '/depth.jpg'
            ir_path = video_name + '/ir.jpg'

            HOG_ir_path = video_name + '/HOG_ir.jpg'
            PLGF_ir_path = video_name + '/PLGF_ir.jpg'

            rgb_path = os.path.join(self.root_dir_dict['USC'], rgb_path)
            depth_path = os.path.join(self.root_dir_dict['USC'], depth_path)
            ir_path = os.path.join(self.root_dir_dict['USC'], ir_path)

            HOG_ir_path = os.path.join(self.root_dir_dict['USC'], HOG_ir_path)
            PLGF_ir_path = os.path.join(self.root_dir_dict['USC'], PLGF_ir_path)
            spoofing_label = landmarks_frame.iloc[idx, 1]

            if random.uniform(0.0, 1.0) <= self.drop_modal_rate:
                depth_path = None
                ir_path = None
                HOG_ir_path = None
                PLGF_ir_path = None

            if missing_modal is not None:
                if 'D' in missing_modal:
                    if random.uniform(0.0, 1.0) <= self.drop_modal_rate:
                        depth_path = None
                if 'IR' in missing_modal:
                    if random.uniform(0.0, 1.0) <= self.drop_modal_rate:
                        ir_path = None
                        HOG_ir_path = None
                        PLGF_ir_path = None
            if self.get_type == 'pos' and spoofing_label == 0:
                continue
            if self.get_type == 'neg' and spoofing_label == 1:
                continue

            modality_domain = 'USC'
            if 'D' in missing_modal and 'IR' not in missing_modal:
                modality_domain = 'USC_no_D'
            if 'IR' in missing_modal and 'D' not in missing_modal:
                modality_domain = 'USC_no_IR'
            if 'D' in missing_modal and 'IR' in missing_modal:
                modality_domain = 'USC_no_D_IR'
            self.dir_list.append({
                "image_x": rgb_path,
                "image_x_depth": depth_path,
                "image_x_ir": ir_path,
                "image_x_ir_HOG": HOG_ir_path,
                "image_x_ir_PLGF": PLGF_ir_path,
                "spoofing_label": spoofing_label,
                "domain": 'USC' if spoofing_label == 0 else 'True',
                "modality_domain": modality_domain
            })

    def __getitem__(self, idx):
        spoofing_label = self.dir_list[idx]['spoofing_label']
        image_x, map_x1 = self.get_single_image_x_RGB(self.dir_list[idx]['image_x'])
        image_x_depth = self.get_single_image_x(self.dir_list[idx]['image_x_depth'])
        image_x_ir = self.get_single_image_x(self.dir_list[idx]['image_x_ir'])
        image_x_ir_HOG = self.get_single_image_x(self.dir_list[idx]['image_x_ir_HOG'])
        image_x_ir_PLGF = self.get_single_image_x(self.dir_list[idx]['image_x_ir_PLGF'])

        if spoofing_label == 1:  # real
            spoofing_label = 1  # real
        else:  # fake
            spoofing_label = 0
            map_x1 = np.zeros((28, 28))

        sample = {'image_x': image_x, 'image_x_depth': image_x_depth, 'image_x_ir': image_x_ir,
                  'image_x_ir_HOG': image_x_ir_HOG, 'image_x_ir_PLGF': image_x_ir_PLGF, 'map_x1': map_x1,
                  'spoofing_label': spoofing_label}
        try:
            if self.transform:
                sample = self.transform(sample)
        except:
            print(self.dir_list[idx])
            if self.transform:
                sample = self.transform(sample)
        sample['domain'] = self.domain_id[self.dir_list[idx]['domain']]
        sample['modality_domain'] = self.modality_domain_id[self.dir_list[idx]['modality_domain']]
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
        if image_path is None:
            image_x = np.zeros((224, 224, 3))
            return image_x
        else:
            # RGB
            image_x_temp = cv2.imread(image_path)

            # cv2.imwrite('temp.jpg', image_x_temp)

            image_x = cv2.resize(image_x_temp, (224, 224))

            # data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
            image_x_aug = seq.augment_image(image_x)

        return image_x_aug

