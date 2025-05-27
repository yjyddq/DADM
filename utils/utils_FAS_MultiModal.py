import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import sklearn
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import pandas as pd
import pdb
import random
import warnings

warnings.filterwarnings("ignore")


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_HTER(y_true_list, y_pred_list, threshold):
    """
    HTER = (FAR + FRR) / 2
    FAR = FA / NI
    FRR = FR / NC
    FA：假脸被当成真脸的数量
    FR：真脸被当成假脸的数量
    NI：假脸个数
    NC：真脸个数
    :param y_true_list:
    :param y_pred_list:
    :param threshold:
    :return:
    """
    pass


def get_threshold(score_file):
    with open(score_file, 'r') as file:
        lines = file.readlines()

    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        angle = float(tokens[0])
        # pdb.set_trace()
        type = int(tokens[1])
        data.append({'map_score': angle, 'label': type})
        if type == 1:
            num_real += 1
        else:
            num_fake += 1

    min_error = count  # account ACER (or ACC)
    min_threshold = 0.0
    min_ACC = 0.0
    min_ACER = 0.0
    min_APCER = 0.0
    min_BPCER = 0.0

    for d in data:
        threshold = d['map_score']

        type1 = len([s for s in data if s['map_score'] <= threshold and s['label'] == 1])
        type2 = len([s for s in data if s['map_score'] > threshold and s['label'] == 0])

        ACC = 1 - (type1 + type2) / count
        APCER = type2 / num_fake
        BPCER = type1 / num_real
        ACER = (APCER + BPCER) / 2.0

        if ACER < min_error:
            min_error = ACER
            min_threshold = threshold
            min_ACC = ACC
            min_ACER = ACER
            min_APCER = APCER
            min_BPCER = min_BPCER

    # print(min_error, min_threshold)
    return min_threshold, min_ACC, min_APCER, min_BPCER, min_ACER


def test_threshold_based(threshold, score_file):
    with open(score_file, 'r') as file:
        lines = file.readlines()

    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        angle = float(tokens[0])
        type = int(tokens[1])
        data.append({'map_score': angle, 'label': type})
        if type == 1:
            num_real += 1
        else:
            num_fake += 1

    type1 = len([s for s in data if s['map_score'] <= threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > threshold and s['label'] == 0])

    ACC = 1 - (type1 + type2) / count
    APCER = type2 / num_fake
    BPCER = type1 / num_real
    ACER = (APCER + BPCER) / 2.0

    return ACC, APCER, BPCER, ACER


# TPR@FPR=0.0001
def tpr_fpr00001_funtion(y_true, y_predict):
    d = pd.DataFrame()
    d['prob'] = list(y_predict)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer - 0.0001).idxmin()]
    # TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
    # TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    return TR1


# TPR@FPR=0.001
def tpr_fpr0001_funtion(y_true, y_predict):
    d = pd.DataFrame()
    d['prob'] = list(y_predict)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer - 0.001).idxmin()]
    # TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
    # TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    return TR1


# TPR@FPR=0.01
def tpr_fpr001_funtion(y_true, y_predict):
    d = pd.DataFrame()
    d['prob'] = list(y_predict)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer - 0.01).idxmin()]
    # TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
    # TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    return TR1


def performances_FAS_MultiModal(CASIA_SURF_CeFA_val_filename, CASIA_SURF_CeFA_test_filename, WMCA_test_filename,
                                WMCA_test_bonafide_filename, WMCA_test_fakehead_filename,
                                WMCA_test_flexiblemask_filename, WMCA_test_glasses_filename,
                                WMCA_test_papermask_filename, WMCA_test_print_filename, WMCA_test_replay_filename,
                                WMCA_test_rigidmask_filename):
    # val
    with open(CASIA_SURF_CeFA_val_filename, 'r') as file1:
        lines = file1.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])  # label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    fpr, tpr, threshold = roc_curve(val_labels, val_scores, pos_label=1)
    val_err, val_threshold = get_err_threhold(fpr, tpr, threshold)

    ################################################
    # intra-test    ACER_CASIA_SURF_CeFA
    with open(CASIA_SURF_CeFA_test_filename, 'r') as file2:
        lines = file2.readlines()
    test_scores = []
    test_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])  # label = int(tokens[1])
        test_scores.append(score)
        test_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    # test based on val_threshold
    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])

    test_APCER = type2 / num_fake
    test_BPCER = type1 / num_real
    ACER_CASIA_SURF_CeFA = (test_APCER + test_BPCER) / 2.0

    # test based on test_threshold
    fpr_test, tpr_test, threshold_test = roc_curve(test_labels, test_scores, pos_label=1)
    err_test, best_test_threshold = get_err_threhold(fpr_test, tpr_test, threshold_test)

    type1 = len([s for s in data if s['map_score'] <= best_test_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > best_test_threshold and s['label'] == 0])

    test_threshold_APCER = type2 / num_fake
    test_threshold_BPCER = type1 / num_real
    ACER_CASIA_SURF_CeFA_testBest = (test_threshold_APCER + test_threshold_BPCER) / 2.0

    # TPR_FPR0001
    TPR_FPR0001 = tpr_fpr0001_funtion(test_labels, test_scores)

    ################################################
    # cross-test    WMCA

    val_threshold = 0.5

    with open(WMCA_test_filename, 'r') as file3:
        lines = file3.readlines()
    test_scores = []
    test_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])  # label = int(tokens[1])
        test_scores.append(score)
        test_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    # test based on val_threshold
    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])

    test_APCER = type2 / num_fake
    test_BPCER = type1 / num_real
    ACER_WMCA = (test_APCER + test_BPCER) / 2.0

    # test based on test_threshold
    fpr_test, tpr_test, threshold_test = roc_curve(test_labels, test_scores, pos_label=1)
    err_test, best_test_threshold = get_err_threhold(fpr_test, tpr_test, threshold_test)

    type1 = len([s for s in data if s['map_score'] <= best_test_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > best_test_threshold and s['label'] == 0])

    test_threshold_APCER = type2 / num_fake
    test_threshold_BPCER = type1 / num_real
    ACER_WMCA_testBest = (test_threshold_APCER + test_threshold_BPCER) / 2.0

    # TPR_FPR001
    TPR_FPR001 = tpr_fpr001_funtion(test_labels, test_scores)

    ################################################
    # cross-test    bonafide
    with open(WMCA_test_bonafide_filename, 'r') as file4:
        lines = file4.readlines()
    test_scores = []
    test_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])  # label = int(tokens[1])
        test_scores.append(score)
        test_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    # test based on val_threshold
    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    BPCER_bonafide = type1 / num_real

    # cross-test    fakehead
    with open(WMCA_test_fakehead_filename, 'r') as file5:
        lines = file5.readlines()
    test_scores = []
    test_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])  # label = int(tokens[1])
        test_scores.append(score)
        test_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    # test based on val_threshold
    type2 = len([s for s in data if s['map_score'] > best_test_threshold and s['label'] == 0])
    APCER_fakehead = type2 / num_fake

    # cross-test    flexiblemask
    with open(WMCA_test_flexiblemask_filename, 'r') as file6:
        lines = file6.readlines()
    test_scores = []
    test_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])  # label = int(tokens[1])
        test_scores.append(score)
        test_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    # test based on val_threshold
    type2 = len([s for s in data if s['map_score'] > best_test_threshold and s['label'] == 0])
    APCER_flexiblemask = type2 / num_fake

    # cross-test    glasses
    with open(WMCA_test_glasses_filename, 'r') as file7:
        lines = file7.readlines()
    test_scores = []
    test_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])  # label = int(tokens[1])
        test_scores.append(score)
        test_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    # test based on val_threshold
    type2 = len([s for s in data if s['map_score'] > best_test_threshold and s['label'] == 0])
    APCER_glasses = type2 / num_fake

    # cross-test    papermask
    with open(WMCA_test_papermask_filename, 'r') as file8:
        lines = file8.readlines()
    test_scores = []
    test_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])  # label = int(tokens[1])
        test_scores.append(score)
        test_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    # test based on val_threshold
    type2 = len([s for s in data if s['map_score'] > best_test_threshold and s['label'] == 0])
    APCER_papermask = type2 / num_fake

    # cross-test    print
    with open(WMCA_test_print_filename, 'r') as file9:
        lines = file9.readlines()
    test_scores = []
    test_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])  # label = int(tokens[1])
        test_scores.append(score)
        test_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    # test based on val_threshold
    type2 = len([s for s in data if s['map_score'] > best_test_threshold and s['label'] == 0])
    APCER_print = type2 / num_fake

    # cross-test    replay
    with open(WMCA_test_replay_filename, 'r') as file10:
        lines = file10.readlines()
    test_scores = []
    test_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])  # label = int(tokens[1])
        test_scores.append(score)
        test_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    # test based on val_threshold
    type2 = len([s for s in data if s['map_score'] > best_test_threshold and s['label'] == 0])
    APCER_replay = type2 / num_fake

    # cross-test    rigidmask
    with open(WMCA_test_rigidmask_filename, 'r') as file11:
        lines = file11.readlines()
    test_scores = []
    test_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])  # label = int(tokens[1])
        test_scores.append(score)
        test_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    # test based on val_threshold
    type2 = len([s for s in data if s['map_score'] > best_test_threshold and s['label'] == 0])
    APCER_rigidmask = type2 / num_fake

    # return ACER_CASIA_SURF_CeFA, ACER_CASIA_SURF_CeFA_testBest, TPR_FPR0001, ACER_WMCA, ACER_WMCA_testBest, TPR_FPR001, BPCER_bonafide, APCER_fakehead, APCER_flexiblemask, APCER_glasses, APCER_papermask, APCER_print, APCER_replay,  APCER_rigidmask
    return ACER_CASIA_SURF_CeFA, ACER_CASIA_SURF_CeFA_testBest, TPR_FPR0001, ACER_WMCA, test_APCER, test_BPCER, ACER_WMCA_testBest, TPR_FPR001, BPCER_bonafide, APCER_fakehead, APCER_flexiblemask, APCER_glasses, APCER_papermask, APCER_print, APCER_replay, APCER_rigidmask


def performances_FAS_MmFA_WMCA(CASIA_SURF_CeFA_val_filename, CASIA_SURF_CeFA_test_filename, WMCA_test_filename):
    # val
    with open(CASIA_SURF_CeFA_val_filename, 'r') as file1:
        lines = file1.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])  # label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    fpr, tpr, threshold = roc_curve(val_labels, val_scores, pos_label=1)
    val_err, val_threshold = get_err_threhold(fpr, tpr, threshold)

    ################################################
    # intra-test    ACER_CASIA_SURF_CeFA
    with open(CASIA_SURF_CeFA_test_filename, 'r') as file2:
        lines = file2.readlines()
    test_scores = []
    test_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])  # label = int(tokens[1])
        test_scores.append(score)
        test_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    # test based on val_threshold
    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])

    test_APCER = type2 / num_fake
    test_BPCER = type1 / num_real
    ACER_CASIA_SURF_CeFA = (test_APCER + test_BPCER) / 2.0

    # test based on test_threshold
    fpr_test, tpr_test, threshold_test = roc_curve(test_labels, test_scores, pos_label=1)
    err_test, best_test_threshold = get_err_threhold(fpr_test, tpr_test, threshold_test)

    type1 = len([s for s in data if s['map_score'] <= best_test_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > best_test_threshold and s['label'] == 0])

    test_threshold_APCER = type2 / num_fake
    test_threshold_BPCER = type1 / num_real
    ACER_CASIA_SURF_CeFA_testBest = (test_threshold_APCER + test_threshold_BPCER) / 2.0

    # TPR_FPR0001
    TPR_FPR00001 = tpr_fpr00001_funtion(test_labels, test_scores)

    ################################################
    # cross-test    WMCA

    val_threshold = 0.5

    with open(WMCA_test_filename, 'r') as file3:
        lines = file3.readlines()
    test_scores = []
    test_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])  # label = int(tokens[1])
        test_scores.append(score)
        test_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    # test based on test_threshold
    fpr_test, tpr_test, threshold_test = roc_curve(test_labels, test_scores, pos_label=1)
    err_test, best_test_threshold = get_err_threhold(fpr_test, tpr_test, threshold_test)

    type1 = len([s for s in data if s['map_score'] <= best_test_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > best_test_threshold and s['label'] == 0])

    test_threshold_APCER = type2 / num_fake
    test_threshold_BPCER = type1 / num_real
    ACER_WMCA_testBest = (test_threshold_APCER + test_threshold_BPCER) / 2.0

    return test_APCER, test_BPCER, ACER_CASIA_SURF_CeFA, test_threshold_APCER, test_threshold_BPCER, ACER_CASIA_SURF_CeFA_testBest, TPR_FPR00001, ACER_WMCA_testBest


def performances_WMCA(WMCA_val_filename, WMCA_test_filename):
    # val
    with open(WMCA_val_filename, 'r') as file1:
        lines = file1.readlines()
    val_scores = []
    val_live_scores = []
    val_attack_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])  # label = int(tokens[1])
        if label == 1:
            num_real += 1
            val_live_scores.append(score)
        else:
            num_fake += 1
            val_attack_scores.append(score)

    scores_dev_live = sorted(val_live_scores)
    th_bpcer = scores_dev_live[int(len(scores_dev_live) * 0.01)]

    # test
    val_threshold = th_bpcer

    with open(WMCA_test_filename, 'r') as file3:
        lines = file3.readlines()
    test_scores = []
    test_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])  # label = int(tokens[1])
        test_scores.append(score)
        test_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    # test based on val_threshold
    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])

    test_APCER = type2 / num_fake
    test_BPCER = type1 / num_real
    ACER_WMCA = (test_APCER + test_BPCER) / 2.0

    # test based on test_threshold
    fpr_test, tpr_test, threshold_test = roc_curve(test_labels, test_scores, pos_label=1)
    err_test, best_test_threshold = get_err_threhold(fpr_test, tpr_test, threshold_test)

    type1 = len([s for s in data if s['map_score'] <= best_test_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > best_test_threshold and s['label'] == 0])

    test_threshold_APCER = type2 / num_fake
    test_threshold_BPCER = type1 / num_real
    ACER_WMCA_testBest = (test_threshold_APCER + test_threshold_BPCER) / 2.0
    # print(val_threshold)
    return test_APCER, test_BPCER, ACER_WMCA, ACER_WMCA_testBest


def get_err_threhold(fpr, tpr, threshold):
    RightIndex = (tpr + (1 - fpr) - 1);
    right_index = np.argmax(RightIndex)
    best_th = threshold[right_index]
    err = fpr[right_index]

    differ_tpr_fpr_1 = tpr + fpr - 1.0

    right_index = np.argmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    err = fpr[right_index]

    # print(err, best_th)
    return err, best_th


# def performances(dev_scores, dev_labels, test_scores, test_labels):
def performances_FAS_Separate(SiW_test_filename, test_3DMAD_filename, HKBU_test_filename, MSU_test_filename,
                              test_3DMask_filename):
    # SiW
    with open(SiW_test_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    fpr, tpr, threshold = roc_curve(val_labels, val_scores, pos_label=1)
    EER_SiW, val_threshold = get_err_threhold(fpr, tpr, threshold)

    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])

    # val_ACC = 1-(type1 + type2) / count
    AUC_SiW = auc(fpr, tpr)
    val_APCER = type2 / num_fake
    val_BPCER = type1 / num_real
    ACER_SiW = (val_APCER + val_BPCER) / 2.0

    # 3DMAD
    with open(test_3DMAD_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    fpr, tpr, threshold = roc_curve(val_labels, val_scores, pos_label=1)
    EER_3DMAD, val_threshold = get_err_threhold(fpr, tpr, threshold)

    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])

    # val_ACC = 1-(type1 + type2) / count
    AUC_3DMAD = auc(fpr, tpr)
    val_APCER = type2 / num_fake
    val_BPCER = type1 / num_real
    ACER_3DMAD = (val_APCER + val_BPCER) / 2.0

    # HKBU
    with open(HKBU_test_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    fpr, tpr, threshold = roc_curve(val_labels, val_scores, pos_label=1)
    EER_HKBU, val_threshold = get_err_threhold(fpr, tpr, threshold)

    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])

    # val_ACC = 1-(type1 + type2) / count
    AUC_HKBU = auc(fpr, tpr)
    val_APCER = type2 / num_fake
    val_BPCER = type1 / num_real
    ACER_HKBU = (val_APCER + val_BPCER) / 2.0

    # MSU
    with open(MSU_test_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    fpr, tpr, threshold = roc_curve(val_labels, val_scores, pos_label=1)
    EER_MSU, val_threshold = get_err_threhold(fpr, tpr, threshold)

    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])

    # val_ACC = 1-(type1 + type2) / count
    AUC_MSU = auc(fpr, tpr)
    val_APCER = type2 / num_fake
    val_BPCER = type1 / num_real
    ACER_MSU = (val_APCER + val_BPCER) / 2.0

    # 3DMask
    with open(test_3DMask_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    fpr, tpr, threshold = roc_curve(val_labels, val_scores, pos_label=1)
    EER_3DMask, val_threshold = get_err_threhold(fpr, tpr, threshold)

    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])

    # val_ACC = 1-(type1 + type2) / count
    AUC_3DMask = auc(fpr, tpr)
    val_APCER = type2 / num_fake
    val_BPCER = type1 / num_real
    ACER_3DMask = (val_APCER + val_BPCER) / 2.0

    # return val_threshold, best_test_threshold, val_ACC, val_ACER, test_ACC, test_APCER, test_BPCER, test_ACER, test_threshold_ACER
    return AUC_SiW, EER_SiW, ACER_SiW, AUC_3DMAD, EER_3DMAD, ACER_3DMAD, AUC_HKBU, EER_HKBU, ACER_HKBU, AUC_MSU, EER_MSU, ACER_MSU, AUC_3DMask, EER_3DMask, ACER_3DMask


# def performances(dev_scores, dev_labels, test_scores, test_labels):
def performances_Deepfake_Separate(FF_test_filename, DFDC_test_filename, test_CelebDF_filename):
    # FF
    with open(FF_test_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    fpr, tpr, threshold = roc_curve(val_labels, val_scores, pos_label=1)
    EER_FF, val_threshold = get_err_threhold(fpr, tpr, threshold)

    # val_ACC = 1-(type1 + type2) / count
    AUC_FF = auc(fpr, tpr)

    # DFDC
    with open(DFDC_test_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    fpr, tpr, threshold = roc_curve(val_labels, val_scores, pos_label=1)
    EER_DFDC, val_threshold = get_err_threhold(fpr, tpr, threshold)

    # val_ACC = 1-(type1 + type2) / count
    AUC_DFDC = auc(fpr, tpr)

    # CelebDF
    with open(test_CelebDF_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    fpr, tpr, threshold = roc_curve(val_labels, val_scores, pos_label=1)
    EER_CelebDF, val_threshold = get_err_threhold(fpr, tpr, threshold)

    # val_ACC = 1-(type1 + type2) / count
    AUC_CelebDF = auc(fpr, tpr)

    # return val_threshold, best_test_threshold, val_ACC, val_ACER, test_ACC, test_APCER, test_BPCER, test_ACER, test_threshold_ACER
    return AUC_FF, EER_FF, AUC_DFDC, EER_DFDC, AUC_CelebDF, EER_CelebDF


def performances_SiW_EER(map_score_val_filename):
    # val
    with open(map_score_val_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    fpr, tpr, threshold = roc_curve(val_labels, val_scores, pos_label=1)
    val_err, val_threshold = get_err_threhold(fpr, tpr, threshold)

    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])

    val_ACC = 1 - (type1 + type2) / count
    val_APCER = type2 / num_fake
    val_BPCER = type1 / num_real
    val_ACER = (val_APCER + val_BPCER) / 2.0

    return val_threshold, val_ACC, val_APCER, val_BPCER, val_ACER


def performances_SiWM_EER(map_score_val_filename):
    # val
    with open(map_score_val_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    fpr, tpr, threshold = roc_curve(val_labels, val_scores, pos_label=1)
    val_err, val_threshold = get_err_threhold(fpr, tpr, threshold)

    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])

    val_ACC = 1 - (type1 + type2) / count
    val_APCER = type2 / num_fake
    val_BPCER = type1 / num_real
    val_ACER = (val_APCER + val_BPCER) / 2.0

    return val_threshold, val_err, val_ACC, val_APCER, val_BPCER, val_ACER


def get_err_threhold_CASIA_Replay(fpr, tpr, threshold):
    RightIndex = (tpr + (1 - fpr) - 1);
    right_index = np.argmax(RightIndex)
    best_th = threshold[right_index]
    err = fpr[right_index]

    differ_tpr_fpr_1 = tpr + fpr - 1.0

    right_index = np.argmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    err = fpr[right_index]

    # print(err, best_th)
    return err, best_th, right_index


def performances_CASIA_Replay(map_score_val_filename):
    # val
    with open(map_score_val_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])  # int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    fpr, tpr, threshold = roc_curve(val_labels, val_scores, pos_label=1)
    val_err, val_threshold, right_index = get_err_threhold_CASIA_Replay(fpr, tpr, threshold)

    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])

    val_ACC = 1 - (type1 + type2) / count

    FRR = 1 - tpr  # FRR = 1 - TPR

    HTER = (fpr + FRR) / 2.0  # error recognition rate &  reject recognition rate

    # return val_ACC, fpr[right_index], FRR[right_index], HTER[right_index]
    return HTER[right_index]


def performances_ZeroShot(map_score_val_filename):
    # val
    with open(map_score_val_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    fpr, tpr, threshold = roc_curve(val_labels, val_scores, pos_label=1)
    auc_val = metrics.auc(fpr, tpr)

    val_err, val_threshold, right_index = get_err_threhold_CASIA_Replay(fpr, tpr, threshold)

    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])

    val_ACC = 1 - (type1 + type2) / count

    FRR = 1 - tpr  # FRR = 1 - TPR

    HTER = (fpr + FRR) / 2.0  # error recognition rate &  reject recognition rate

    return val_ACC, auc_val, HTER[right_index]


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
