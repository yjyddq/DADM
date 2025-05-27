from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
import argparse, os
import pandas as pd
import cv2
import numpy as np
import random
import math
import json
import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from data_preprocess.Load_FAS_MultiModal import Normaliztion, ToTensor, RandomHorizontalFlip
from data_preprocess.Load_multi_domain_multi_modal import Multi_Domain_Spoofing_train, Flexible_Multi_Domain_Spoofing_train
from models.model_factory import get_model
from step import step_batch, step_batch_regrad, perform_eval_multi_domain
from utils.common_util import forward_model, CosineAnnealingLR_with_Restart
from utils.gpu_mem_track import MemTracker
from utils.utils_FAS_MultiModal import AvgrageMeter, setup_seed
from utils.my_loss import SingleSideCELoss, AngleLoss
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import copy
import pdb
import time
from regrad import calculate_prototype



##########    Dataset root    ##########
def FeatureMap2Heatmap(x, x2, x3, dir_name):
    ## initial images
    org_img = x[0, :, :, :].cpu()
    org_img = org_img.data.numpy() * 128 + 127.5
    org_img = org_img.transpose((1, 2, 0))
    org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(dir_name + '/x_visual.jpg', org_img)

    org_img = x2[0, :, :, :].cpu()
    org_img = org_img.data.numpy() * 128 + 127.5
    org_img = org_img.transpose((1, 2, 0))
    org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(dir_name + '/x_depth.jpg', org_img)

    org_img = x3[0, :, :, :].cpu()
    org_img = org_img.data.numpy() * 128 + 127.5
    org_img = org_img.transpose((1, 2, 0))
    org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(dir_name + '/x_ir.jpg', org_img)


# main function
def train_test(args):
    # 初始化随机种子
    setup_seed(args.seed)
    """Load logs"""
    base_dir = '/home/young/DADM/logs'
    log_dir = os.path.join(base_dir, f"{args.model}_{args.modality}", str(int(time.time() * 10000)))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_name = os.path.join(log_dir, 'main_log.txt')
    log_file = open(log_name, 'w')

    for arg in vars(args):
        log_file.write(f"{arg}: {getattr(args, arg)}\n")

    """Load model"""
    model = get_model(args.model, args)
    model = model.cuda()
    
    # Check the states of frozen params
    # for n, param in model.named_parameters():
    #     print(n, param.requires_grad)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00005)
    # scheduler = CosineAnnealingLR_with_Restart(optim, T_max=30, T_mult=1, eta_min=5e-7)

    test_out_filename = log_name.replace('_log.txt', '_out_test.txt')

    criterion = nn.CrossEntropyLoss()
    best_HTER, best_epoch = 1.0, -1,

    # Include below two losses when usign regrad
    domain_prototypes = (None, None, None)
    ssp_criterion = SingleSideCELoss()
    ang_criterion = AngleLoss()

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        lr = args.lr
        loss_absolute = AvgrageMeter()
        loss_contra = AvgrageMeter()
        loss_absolute_RGB = AvgrageMeter()

        ###########################################
        '''                train                '''
        ###########################################
        if not args.missing_modality:
            train_data = Multi_Domain_Spoofing_train(args.train, transforms.Compose([RandomHorizontalFlip(), ToTensor(), Normaliztion()]))
        else:
            train_data = Flexible_Multi_Domain_Spoofing_train(args.train, transforms.Compose([RandomHorizontalFlip(), ToTensor(), Normaliztion()]))

        dataloader_train = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=3)

        model.train()
        n_cnt = 0
        for i, sample_batched in enumerate(tqdm.tqdm(dataloader_train)):
            """training without regrad
            _, loss = step_batch(model, sample_batched, optim, args.modality, 'train', criterion=criterion)
            """
            # if i == 30:
            #     break
            """
            # training with regrad"""
            domain_prototypes = calculate_prototype(
                model, sample_batched, epoch,
                domain_prototypes[0], domain_prototypes[1], domain_prototypes[2],
                sample_scale=1.0, momentum_coef=0.2,
                n_classes=5, use_spoof_type=False,
                embed_dim=768
            )
            _, loss = step_batch_regrad(model, sample_batched, optim, args.modality, 'train', criterion=criterion,
                                        ssp_criterion=ssp_criterion, ang_criterion=ang_criterion, prototypes=domain_prototypes)

            _ = model.update_hyperplane()
            # gpu_tracker.track()
            n = sample_batched['image_x'].shape[0]
            # print(loss)
            if isinstance(loss, dict):
                loss_absolute.update(loss['total'].data, n)
                loss_contra.update(loss['total'].data, n)
                loss_absolute_RGB.update(loss['total'].data, n)
            else:
                loss_absolute.update(loss.data, n)
                loss_contra.update(loss.data, n)
                loss_absolute_RGB.update(loss.data, n)
            n_cnt += n

        log_file.write('epoch:%d, mini-batch:%3d, lr=%f, CE_global= %.6f , CE1= %.6f , CE2= %.6f \n' % (
            epoch + 1, i + 1, lr, loss_absolute.avg, loss_contra.avg, loss_absolute_RGB.avg))
        log_file.flush()
        ###########################################
        '''                test                 '''
        ###########################################
        best_HTER, best_epoch = perform_eval_multi_domain(
            model, optim, args,
            test_out_filename, epoch, -1, best_HTER,
            best_epoch, log_dir, log_file
        )
    print('Finished Training')
    log_file.close()

    return best_HTER


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--lr', type=float, default=1e-6, help='initial learning rate')  # default=0.0003   0.01
    parser.add_argument('--batchsize', type=int, default=32, help='initial batchsize')  # default=16
    parser.add_argument('--step_size', type=int, default=20, help='how many epochs lr decays once')  # 500  | DPC = 400
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--epochs', type=int, default=80, help='total training epochs')
    parser.add_argument('--model', type=str, default="uc_vit_1modal", help='see model_factory.py')
    parser.add_argument('--save_best', action='store_true', default=True, help='True  -->  save the best weight; False -->  dont save')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--modality', type=str, default="RGB", help='RGB/D/IR/RGBD/RGBIR/RGBDIR')
    parser.add_argument('--missing_modality', type=bool, default=False, help='True/False')
    parser.add_argument('--train', nargs='+', default="WMCA", help='WMCAGT(WMCA ground test)/')
    parser.add_argument('--test', nargs='+', default="WMCA", help='WMCAGT(WMCA ground test)/')
    parser.add_argument('--adapter_dim', type=int, default=8)
    parser.add_argument('--hidden_size', type=int, default=768, help='ViT的hidden size')

    args = parser.parse_args()

    train_test(args)

