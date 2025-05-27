import gc

import torch
import torch.nn as nn
import numpy as np
#import torchsnooper
from torch import optim
import re
# from models.model_factory import get_model
from utils.common_util import forward_model_with_domain
import random


def EU_dist(x1, x2):
    d_matrix = torch.zeros(x1.shape[0], x2.shape[0]).to(x1.device)
    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            d = torch.sqrt(torch.dot((x1[i] - x2[j]), (x1[i] - x2[j])))
            d_matrix[i, j] = d
    return d_matrix

def calculate_prototype(model, sample_batched, epoch, color_proto=None, ir_proto=None,
                        depth_proto=None, sample_scale=0.5, embed_dim=768, momentum_coef=0.2, n_classes=5,
                        use_spoof_type=False):
    """calculate once per epoch"""
    """
    1. Each spoof type in a domain is a class
    2. The live samples of all domains is a class;
    """
    # n_classes = 4 + 1

    color_prototypes = torch.zeros(n_classes, embed_dim).cuda()
    ir_prototypes = torch.zeros(n_classes, embed_dim).cuda()
    depth_prototypes = torch.zeros(n_classes, embed_dim).cuda()
    count_domain = [0 for _ in range(n_classes)]

    # calculate prototype
    model.eval()
    with torch.no_grad():
        # sample_count = 0
        # all_num = len(dataloader)

        # print(sample_batched.keys())
        inputs = sample_batched['image_x'].cuda()
        inputs_depth = sample_batched['image_x_depth'].cuda()
        inputs_ir = sample_batched['image_x_ir'].cuda()
        inputs_ir_HOG = sample_batched['image_x_ir_HOG'].cuda()
        inputs_ir_PLGF = sample_batched['image_x_ir_PLGF'].cuda()
        if use_spoof_type:
            domain_id = sample_batched['spoofing_label']
        else:
            domain_id = sample_batched['domain']
        inputs_ir = torch.cat(
            (inputs_ir[:, 0].unsqueeze(1), inputs_ir_HOG[:, 0].unsqueeze(1), inputs_ir_PLGF[:, 0].unsqueeze(1)),
            dim=1
        )
        logits = forward_model_with_domain(
            model, inputs, inputs_depth, inputs_ir, 'RGBDIR', sample_batched['domain'], sample_batched['modality_domain']
        )
        for p in range(len(inputs)):
            count_domain[domain_id[p]] += 1
            # print(logits['feat_1'].shape, color_prototypes.shape)
            color_prototypes[domain_id[p], :] += logits['feat_1'][p]
            depth_prototypes[domain_id[p], :] += logits['feat_2'][p]
            ir_prototypes[domain_id[p], :] += logits['feat_3'][p]


    for c in range(n_classes):
        if count_domain[c] == 0:
            continue
        color_prototypes[c, :] /= count_domain[c]
        depth_prototypes[c, :] /= count_domain[c]
        ir_prototypes[c, :] /= count_domain[c]

    if epoch <= 0:
        color_prototypes = color_prototypes
        depth_prototypes = depth_prototypes
        ir_prototypes = ir_prototypes
    else:
        color_prototypes = (1.0 - momentum_coef) * color_prototypes + momentum_coef * color_proto
        depth_prototypes = (1.0 - momentum_coef) * depth_prototypes + momentum_coef * depth_proto
        ir_prototypes = (1.0 - momentum_coef) * ir_prototypes + momentum_coef * ir_proto

    return color_prototypes, depth_prototypes, ir_prototypes

def get_layer_id(l_name):
    match = re.search(r'\d+', l_name.split('.')[0])
    if match:
        first_number = match.group()
        return first_number
    else:
        return None


def cal_orthogonal_grad(base_grad, decompose_grad, eps=1e-8):
    """
    :param base_grad: 基grad
    :param decompose_grad: 要分解的grad
    :param eps: 防止除出nan
    :return:
    """
    grad_norm = torch.dot(base_grad.flatten(), base_grad.flatten()) + eps
    proj_len = torch.dot(base_grad.flatten(), decompose_grad.flatten())
    factor = proj_len / grad_norm
    # print(factor)
    # print(base_grad, decompose_grad, proj_len)
    factor = torch.tensor(0.0) if torch.isnan(factor) else factor
    # if factor < 0.0:
    #     print(factor)
    non_orthogonal_grad = factor * base_grad
    # print(non_orthogonal_grad)
    return non_orthogonal_grad


def cal_same_dir_grad(base_grad, decompose_grad, eps=1e-8):
    """
    获取decompose_grad中与base_grad同向的部分
    :param base_grad: 基grad
    :param decompose_grad: 要分解的grad
    :param eps: 防止除出nan
    :return:
    """
    grad_norm = torch.dot(base_grad.flatten(), base_grad.flatten()) + eps
    proj_len = torch.dot(base_grad.flatten(), decompose_grad.flatten())
    factor = proj_len / grad_norm
    factor = torch.tensor(0.0) if torch.isnan(factor) else factor
    factor = factor if factor >= 0.0 else torch.tensor(0.0)
    non_orthogonal_grad = factor * base_grad

    return non_orthogonal_grad


def delete_conflict_grad(base_grad, decompose_grad, eps=1e-8):
    """
    获取decompose_grad中与base_grad同向的部分
    :param base_grad: 基grad
    :param decompose_grad: 要分解的grad
    :param eps: 防止除出nan
    :return:
    """
    grad_norm = torch.dot(base_grad.flatten(), base_grad.flatten()) + eps
    proj_len = torch.dot(base_grad.flatten(), decompose_grad.flatten())
    factor = proj_len / grad_norm
    factor = torch.tensor(0.0) if torch.isnan(factor) else factor
    orthogonal_grad = decompose_grad - base_grad * factor
    factor = factor if factor >= 0.0 else torch.tensor(0.0)
    non_orthogonal_grad = factor * base_grad

    return orthogonal_grad + non_orthogonal_grad


def get_slow_modal_grad(main_grad, sub_grad_1, sub_grad_2):
    """
    调制慢模态
    :param main_grad:
    :param sub_grad_1:
    :param sub_grad_2:
    :return:
    """
    if torch.dot(main_grad.flatten(), sub_grad_1.flatten()) > 0.0:
        modulated_sub_grad_1 = cal_same_dir_grad(main_grad, sub_grad_1)
    else:
        modulated_sub_grad_1 = delete_conflict_grad(main_grad, sub_grad_1)

    if torch.dot(main_grad.flatten(), sub_grad_2.flatten()) > 0.0:
        modulated_sub_grad_2 = cal_same_dir_grad(main_grad, sub_grad_2)
    else:
        modulated_sub_grad_2 = delete_conflict_grad(main_grad, sub_grad_2)

    return main_grad, modulated_sub_grad_1, modulated_sub_grad_2


def get_fast_modal_grad(main_grad, sub_grad_1, sub_grad_2):
    """
    调制快模态
    :param main_grad:
    :param sub_grad_1:
    :param sub_grad_2:
    :return:
    """
    sub_grad_sum = sub_grad_1 + sub_grad_2
    if torch.dot(main_grad.flatten(), sub_grad_sum.flatten()) > 0.0:
        modulated_main_grad = cal_same_dir_grad(sub_grad_sum, main_grad)
    else:
        modulated_main_grad = delete_conflict_grad(sub_grad_sum, main_grad)

    return modulated_main_grad, sub_grad_1, sub_grad_2


def get_named_parameters_with_grad(model, get_type='name'):
    """
    获取参数/参数的梯度列表，避免拿到没有grad的class_token
    :param model:
    :param get_type:
    :return:
    """
    named_param_list = []
    for layer_name, param in model.named_parameters():
        # if hasattr(param, 'grad_clone'):
        if get_type == 'name':
            named_param_list.append(layer_name)
        elif get_type == 'grad':
            if param.grad is not None:
                named_param_list.append(param.grad.clone())
                # print(layer_name, 'Not None')
            else:
                named_param_list.append(None)

    return named_param_list

'''if use MMDG'''
# def backward_regrad_3_modal_no_leak_uem(model, optimizer, loss_dict):
#     loss_1 = loss_dict['m1']
#     loss_2 = loss_dict['m2']
#     loss_3 = loss_dict['m3']
#     loss_total = loss_dict['total']
#
#     temp_loss_dict = {
#         "total": torch.tensor(loss_total.item()),
#         "m1": torch.tensor(loss_1.item()),
#         "m2": torch.tensor(loss_2.item()),
#         "m3": torch.tensor(loss_3.item()),
#     }
#
#     optimizer.zero_grad()
#     loss_1.backward(retain_graph=True)
#     grad_l1_list = get_named_parameters_with_grad(model, 'grad')
#
#     optimizer.zero_grad()
#     loss_2.backward(retain_graph=True)
#     grad_l2_list = get_named_parameters_with_grad(model, 'grad')
#
#     optimizer.zero_grad()
#     loss_3.backward(retain_graph=True)
#     grad_l3_list = get_named_parameters_with_grad(model, 'grad')
#
#     optimizer.zero_grad()
#     loss_total.backward(retain_graph=False)  # 最后一个loss，不需要维护retain_graph
#
#     # print('no_leak memory_before!!: ', torch.cuda.memory_allocated())
#     i = 0
#     for layer_name, param in model.named_parameters():
#         g1 = grad_l1_list[i]
#         g2 = grad_l2_list[i]
#         g3 = grad_l3_list[i]
#         if g1 is None and g2 is None and g3 is None:
#             continue
#         elif get_layer_id(layer_name) == '3':
#             if min(loss_1, loss_2, loss_3) != loss_3:
#                 c_g3, c_g1, c_g2 = get_slow_modal_grad(g3, g1, g2)
#                 corrected_grad = c_g3 + c_g1 * loss_dict['uc_1'] + c_g2 * loss_dict['uc_2']
#             else:
#                 c_g3, c_g1, c_g2 = get_fast_modal_grad(g3, g1, g2)
#                 corrected_grad = c_g3 * loss_dict['uc_3'] + c_g1 + c_g2
#
#         elif get_layer_id(layer_name) == '2':
#             if min(loss_1, loss_2, loss_3) != loss_2:
#                 c_g2, c_g1, c_g3 = get_slow_modal_grad(g2, g1, g3)
#                 corrected_grad = c_g2 + c_g1 * loss_dict['uc_1'] + c_g3 * loss_dict['uc_3']
#             else:
#                 c_g2, c_g1, c_g3 = get_fast_modal_grad(g2, g1, g3)
#                 corrected_grad = c_g2 * loss_dict['uc_2'] + c_g1 + c_g3
#
#         elif get_layer_id(layer_name) == '1':
#             if min(loss_1, loss_2, loss_3) != loss_1:
#                 c_g1, c_g2, c_g3 = get_slow_modal_grad(g1, g2, g3)
#                 corrected_grad = c_g1 + c_g2 * loss_dict['uc_2'] + c_g3 * loss_dict['uc_3']
#             else:
#                 c_g1, c_g2, c_g3 = get_fast_modal_grad(g1, g2, g3)
#                 corrected_grad = c_g1 * loss_dict['uc_1'] + c_g2 + c_g3
#         if param.grad is not None:
#             param.grad += corrected_grad
#         elif param.grad is None and corrected_grad is not None:
#             param.grad = corrected_grad.clone()
#         g1, g2, g3, c_g1, c_g2, c_g3, corrected_grad = None, None, None, None, None, None, None
#         i += 1
#
#     optimizer.step()
#     optimizer.zero_grad()
#     # print('no_leak memory_last!!: ', torch.cuda.memory_allocated())
#     return temp_loss_dict

def backward_regrad_3_modal_no_leak(model, optimizer, loss_dict):
    loss_1 = loss_dict['m1']
    loss_2 = loss_dict['m2']
    loss_3 = loss_dict['m3']
    loss_total = loss_dict['total']

    temp_loss_dict = {
        "total": torch.tensor(loss_total.item()),
        "m1": torch.tensor(loss_1.item()),
        "m2": torch.tensor(loss_2.item()),
        "m3": torch.tensor(loss_3.item()),
    }

    optimizer.zero_grad()
    loss_1.backward(retain_graph=True)
    grad_l1_list = get_named_parameters_with_grad(model, 'grad')

    optimizer.zero_grad()
    loss_2.backward(retain_graph=True)
    grad_l2_list = get_named_parameters_with_grad(model, 'grad')

    optimizer.zero_grad()
    loss_3.backward(retain_graph=True)
    grad_l3_list = get_named_parameters_with_grad(model, 'grad')

    optimizer.zero_grad()
    loss_total.backward(retain_graph=False)

    # print('no_leak memory_before!!: ', torch.cuda.memory_allocated())
    i = 0
    for layer_name, param in model.named_parameters():
        g1 = grad_l1_list[i]
        g2 = grad_l2_list[i]
        g3 = grad_l3_list[i]
        if g1 is None and g2 is None and g3 is None:
            continue
        elif 'fc' in layer_name:
            continue
        elif get_layer_id(layer_name) == '3' or get_layer_id(layer_name) == '13' or get_layer_id(layer_name) == '23':
            if max(loss_dict['mi_1'], loss_dict['mi_2'], loss_dict['mi_3']) == loss_dict['mi_3']:
                c_g3, c_g1, c_g2 = get_slow_modal_grad(g3, g1, g2)
                corrected_grad = c_g3 + c_g1 * loss_dict['mi_1'] + c_g2 * loss_dict['mi_2']
            else:
                c_g3, c_g1, c_g2 = get_fast_modal_grad(g3, g1, g2)
                corrected_grad = c_g3 * loss_dict['mi_3'] + c_g1 + c_g2

        elif get_layer_id(layer_name) == '2' or get_layer_id(layer_name) == '12' or get_layer_id(layer_name) == '23':
            if max(loss_dict['mi_1'], loss_dict['mi_2'], loss_dict['mi_3']) == loss_dict['mi_2']:
                c_g2, c_g1, c_g3 = get_slow_modal_grad(g2, g1, g3)
                corrected_grad = c_g2 + c_g1 * loss_dict['mi_1'] + c_g3 * loss_dict['mi_3']
            else:
                c_g2, c_g1, c_g3 = get_fast_modal_grad(g2, g1, g3)
                corrected_grad = c_g2 * loss_dict['mi_2'] + c_g1 + c_g3

        elif get_layer_id(layer_name) == '1' or get_layer_id(layer_name) == '12' or get_layer_id(layer_name) == '13':
            if max(loss_dict['mi_1'], loss_dict['mi_2'], loss_dict['mi_3']) == loss_dict['mi_1']:
                c_g1, c_g2, c_g3 = get_slow_modal_grad(g1, g2, g3)
                corrected_grad = c_g1 + c_g2 * loss_dict['mi_2'] + c_g3 * loss_dict['mi_3']
            else:
                c_g1, c_g2, c_g3 = get_fast_modal_grad(g1, g2, g3)
                corrected_grad = c_g1 * loss_dict['mi_1'] + c_g2 + c_g3
        if param.grad is not None:
            param.grad += corrected_grad
        elif param.grad is None and corrected_grad is not None:
            param.grad = corrected_grad.clone()
        g1, g2, g3, c_g1, c_g2, c_g3, corrected_grad = None, None, None, None, None, None, None
        i += 1

    optimizer.step()
    optimizer.zero_grad()
    # print('no_leak memory_last!!: ', torch.cuda.memory_allocated())
    return temp_loss_dict
