import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from data_preprocess.Load_FAS_MultiModal import Normaliztion, ToTensor
from data_preprocess.Load_multi_domain_multi_modal import Multi_Domain_Spoofing_valtest, Flexible_Multi_Domain_Spoofing_valtest
from utils.common_util import forward_model, forward_model_with_domain
from utils.utils_FAS_MultiModal import performances_ZeroShot
from regrad import backward_regrad_3_modal_no_leak


def step_batch(model, sample_batched, optim, modality='RGBDIR', mode='train', criterion=None):
    # get the inputs
    spoof_label = sample_batched['spoofing_label'].cuda()
    inputs = sample_batched['image_x'].cuda()
    inputs_depth = sample_batched['image_x_depth'].cuda()
    inputs_ir = sample_batched['image_x_ir'].cuda()

    optim.zero_grad()

    # forward
    logits = forward_model(model, inputs, inputs_depth, inputs_ir, modality)
    if mode == 'train':
        # training：compute loss and backward
        loss = model.cal_loss(spoof_label, criterion)
        loss.backward()
        optim.step()
        return logits['out_all'], loss
    else:
        # testing：return results
        return logits['out_all'], None

def step_batch_regrad(model, sample_batched, optim, modality='RGBD', mode='train', criterion=None, ssp_criterion=None, ang_criterion=None, prototypes=None):
    spoof_label = sample_batched['spoofing_label'].cuda()
    inputs = sample_batched['image_x'].cuda()
    inputs_depth = sample_batched['image_x_depth'].cuda()
    inputs_ir = sample_batched['image_x_ir'].cuda()

    logits = forward_model_with_domain(model, inputs, inputs_depth, inputs_ir, modality, sample_batched['domain'], sample_batched['modality_domain'])

    optim.zero_grad()

    if mode == 'train':
        loss = model.cal_loss(spoof_label, criterion, ssp_criterion, ang_criterion, prototypes)
        loss = backward_regrad_3_modal_no_leak(model, optim, loss)
        return logits['out_all'], loss
    else:
        return logits['out_all'], None

def perform_eval_multi_domain(model, optim, args, test_out_filename,
                              epoch, batch, best_HTER, best_epoch,
                              log_dir, log_file):
    model.eval()

    with torch.no_grad():
        ###########################################
        #          cross-domain    test     
        ###########################################
        if not args.missing_modality:
            test_data = Multi_Domain_Spoofing_valtest(args.test, transforms.Compose([Normaliztion(), ToTensor()]))
        else:
            test_data = Flexible_Multi_Domain_Spoofing_valtest(args.test, transforms.Compose([Normaliztion(), ToTensor()]))
        dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)

        map_score_list = []

        # Traverse the datasets
        for i, sample_batched in enumerate(dataloader_test):
            logits, _ = step_batch(model, sample_batched, optim, args.modality, 'test')
            # Traverse a batch
            for test_batch in range(sample_batched['image_x'].shape[0]):
                map_score = 0.0
                map_score += F.softmax(logits)[test_batch][1]
                print(map_score, sample_batched['spoofing_label'])
                map_score_list.append('{} {}\n'.format(map_score, sample_batched['spoofing_label'][test_batch][0]))

        # Log the results
        with open(test_out_filename, 'w') as file:
            file.writelines(map_score_list)

        ##########################################################################
        #       performance measurement for both intra- and inter-testings
        ##########################################################################
        _, test_AUC, test_HTER = performances_ZeroShot(test_out_filename)
        out_info = f"\nepoch:{epoch:02}, batch:{batch}, test_HTER:{test_HTER:.4f}, "f"test_AUC:{test_AUC:.4f}"

        # Log the best performance
        if test_HTER < best_HTER:
            best_HTER = test_HTER
            best_epoch = epoch
            if args.save_best:
                torch.save(model.state_dict(), os.path.join(log_dir, 'best_HTER.pt'))

        print(out_info)
        log_file.write('\n' + out_info)

        best_info = f"\n[!!!] BEST best_HTER:{best_HTER:.4f} epoch:{best_epoch} log:{log_dir}\n"
        print(best_info)
        print(args)

        log_file.write('\n' + best_info)
        log_file.flush()
        return best_HTER, best_epoch