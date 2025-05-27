from functools import partial
from baal.bayesian.dropout import Dropout as MCDropout
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchsnooper
from torch.autograd import Variable
import numpy as np
import copy
import math
from einops import rearrange, repeat
import torchvision
from collections import OrderedDict

normalizer = lambda x: x / (torch.norm(x, dim=-1, keepdim=True) + 1e-10)
class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0,
                                groups=self.conv.groups)

            return out_normal - self.theta * out_diff


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class CDC_Adapter(nn.Module):
    def __init__(self, adapterdim=8, theta=0.7):
        super(CDC_Adapter, self).__init__()

        self.adapter_conv = Conv2d_cd(in_channels=adapterdim, out_channels=adapterdim, kernel_size=3, stride=1,
                                      padding=1, theta=theta)

        # nn.init.xavier_uniform_(self.adapter_conv.weight)

        self.adapter_down = nn.Linear(768, adapterdim)  # equivalent to 1 * 1 Conv
        self.adapter_up = nn.Linear(adapterdim, 768)  # equivalent to 1 * 1 Conv
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = adapterdim

    def forward(self, x):
        B, N, C = x.shape

        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        x_patch = x_down[:, 1:].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch = self.adapter_conv(x_patch)
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)

        # pdb.set_trace()

        x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
        x_cls = self.adapter_conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        return x_up


class my_ViT_3modality_CDC_Adapter(nn.Module):
    def __init__(self, ):
        super(my_ViT_3modality_CDC_Adapter, self).__init__()
        self.num_encoders = 12
        dim = 768

        vit_b_16 = torchvision.models.vit_b_16(pretrained=True)
        for p in vit_b_16.parameters():
            p.requires_grad = True

        # extract encoder alone and discard CNN (patchify + linear projection) feature extractor, classifer head
        # Refer Encoder() class in https://pytorch.org/vision/main/_modules/torchvision/models/vision_transformer.html

        self.conv_proj1 = vit_b_16.conv_proj
        for p in self.conv_proj1.parameters():
            p.requires_grad = True
        self.conv_proj2 = copy.deepcopy(self.conv_proj1)
        self.conv_proj3 = copy.deepcopy(self.conv_proj1)

        self.class_token1 = vit_b_16.class_token
        self.class_token2 = copy.deepcopy(self.class_token1)
        self.class_token3 = copy.deepcopy(self.class_token1)

        vit = vit_b_16.encoder

        # self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        seq_length = 14 * 14 + 1

        self.pos_embedding1 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))
        self.pos_embedding2 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))
        self.pos_embedding3 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))

        # start building ViT encoder layers
        layer_list = [vit.dropout]
        for i in range(self.num_encoders):
            if i < (self.num_encoders - 1):
                for p in vit.layers[i].parameters():
                    p.requires_grad = False
            layer_list.append(vit.layers[i])
        # add final encoder layer norm
        layer_list.append(vit.ln)

        # assign models for forward pass
        self.ViT_Encoder = nn.Sequential(*layer_list)
        self.fc_all = nn.Sequential(
            nn.Linear(dim + dim + dim, 2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(dim, 2),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(dim, 2),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(dim, 2),
        )
        self.adapter_1_1, self.adapter_1_2 = [], []
        self.adapter_2_1, self.adapter_2_2 = [], []
        self.adapter_3_1, self.adapter_3_2 = [], []

        for i in range(self.num_encoders):
            self.adapter_1_1.append(CDC_Adapter())
            self.adapter_1_2.append(CDC_Adapter())
            self.adapter_2_1.append(CDC_Adapter())
            self.adapter_2_2.append(CDC_Adapter())
            self.adapter_3_1.append(CDC_Adapter())
            self.adapter_3_2.append(CDC_Adapter())

        self.adapter_1_1 = nn.Sequential(*self.adapter_1_1)
        self.adapter_2_1 = nn.Sequential(*self.adapter_2_1)
        self.adapter_3_1 = nn.Sequential(*self.adapter_3_1)
        self.adapter_1_2 = nn.Sequential(*self.adapter_1_2)
        self.adapter_2_2 = nn.Sequential(*self.adapter_2_2)
        self.adapter_3_2 = nn.Sequential(*self.adapter_3_2)

        self.out = None


    # @torchsnooper.snoop()
    def forward(self, x1, x2, x3):

        b, c, fh, fw = x1.shape
        x1 = self.conv_proj1(x1)  # b,d,gh,gw
        x1 = x1.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x2 = self.conv_proj2(x2)  # b,d,gh,gw
        x2 = x2.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x3 = self.conv_proj3(x3)  # b,d,gh,gw
        x3 = x3.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x1 = torch.cat((self.class_token1.expand(b, -1, -1), x1), dim=1)  # b,gh*gw+1,d
        x2 = torch.cat((self.class_token2.expand(b, -1, -1), x2), dim=1)  # b,gh*gw+1,d
        x3 = torch.cat((self.class_token3.expand(b, -1, -1), x3), dim=1)  # b,gh*gw+1,d

        proj1 = x1 + self.pos_embedding1
        proj2 = x2 + self.pos_embedding2
        proj3 = x3 + self.pos_embedding3

        proj1 = self.ViT_Encoder[0](proj1)
        proj2 = self.ViT_Encoder[0](proj2)
        proj3 = self.ViT_Encoder[0](proj3)

        for i in range(1, min(len(self.ViT_Encoder), len(self.ViT_Encoder)) - 1):
            # 遍历ViT每一层，添加adapter
            x1 = self.ViT_Encoder[i].ln_1(proj1)
            x2 = self.ViT_Encoder[i].ln_1(proj2)
            x3 = self.ViT_Encoder[i].ln_1(proj3)

            x1_c, _ = self.ViT_Encoder[i].self_attention(x1, x1, x1, need_weights=False)
            x2_c, _ = self.ViT_Encoder[i].self_attention(x2, x2, x2, need_weights=False)
            x3_c, _ = self.ViT_Encoder[i].self_attention(x3, x3, x3, need_weights=False)

            x1_c = self.ViT_Encoder[i].dropout(x1_c)
            x2_c = self.ViT_Encoder[i].dropout(x2_c)
            x3_c = self.ViT_Encoder[i].dropout(x3_c)

            x1 = proj1 + x1_c + self.ViT_Encoder[i].ln_1(self.adapter_1_1[i - 1](x1))
            x2 = proj2 + x2_c + self.ViT_Encoder[i].ln_1(self.adapter_2_1[i - 1](x2))
            x3 = proj3 + x3_c + self.ViT_Encoder[i].ln_1(self.adapter_3_1[i - 1](x3))

            y1 = self.ViT_Encoder[i].ln_2(x1)
            y2 = self.ViT_Encoder[i].ln_2(x2)
            y3 = self.ViT_Encoder[i].ln_2(x3)

            y1_m = self.ViT_Encoder[i].mlp(y1)
            y2_m = self.ViT_Encoder[i].mlp(y2)
            y3_m = self.ViT_Encoder[i].mlp(y3)

            proj1 = x1 + y1_m + self.ViT_Encoder[i].ln_2(self.adapter_1_2[i - 1](y1))
            proj2 = x2 + y2_m + self.ViT_Encoder[i].ln_2(self.adapter_2_2[i - 1](y2))
            proj3 = x3 + y3_m + self.ViT_Encoder[i].ln_2(self.adapter_3_2[i - 1](y3))


        proj1 = self.ViT_Encoder[-1](proj1)
        proj2 = self.ViT_Encoder[-1](proj2)
        proj3 = self.ViT_Encoder[-1](proj3)

        logits1 = proj1[:, 0]  # b,d
        out1 = self.fc1(logits1)  # b,num_classes
        logits2 = proj2[:, 0]  # b,d
        out2 = self.fc2(logits2)  # b,num_classes
        logits3 = proj3[:, 0]  # b,d
        out3 = self.fc3(logits3)  # b,num_classes
        logits_all = torch.cat([logits1, logits2, logits3], dim=1)
        out_all = self.fc_all(logits_all)
        self.out = {
            "out_all": out_all,
            "out_1": out1,
            "out_2": out2,
            "out_3": out3
        }
        return self.out

    def cal_loss(self, spoof_label, loss_func):
        loss_global = loss_func(self.out['out_all'], spoof_label.squeeze(-1))
        loss_1 = loss_func(self.out['out_1'], spoof_label.squeeze(-1))
        loss_2 = loss_func(self.out['out_2'], spoof_label.squeeze(-1))
        loss_3 = loss_func(self.out['out_3'], spoof_label.squeeze(-1))
        loss = loss_global + loss_1 + loss_2 + loss_3
        return loss


def cal_uncertainty(attn, dropout, exp_rate, num_sample=20):
    with torch.no_grad():
        sample_list = []
        for j in range(num_sample):
            attn_sample = dropout(attn).requires_grad_(False) # [B, L, N]
            sample_list.append(attn_sample)
        # print(torch.cuda.max_memory_allocated())
        sample_list = torch.stack(sample_list).requires_grad_(False) # [M, B, L, N]
        sample_std = torch.std(sample_list, dim=0, unbiased=True).requires_grad_(False)
        uncertainty = torch.exp(-sample_std * exp_rate).requires_grad_(False)
        sample_list = None
        return uncertainty


class My_MHA_UC(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        # self.fused_attn = use_fused_attn()

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    # @torchsnooper.snoop()
    def forward(self, x_q, x_k, x_v, x_u):
        B, N, C = x_q.shape
        q = self.q_linear(x_q).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x_k).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x_v).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        uc_map = x_u.mean(dim=-1, keepdim=True).unsqueeze(1).expand(B, 1, attn.shape[-1], attn.shape[-1]).cuda()
        attn = attn * uc_map
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class U_Adapter(nn.Module):
    def __init__(self, adapter_dim=8, theta=0.5, use_cdc=True, dropout_rate=0.2, hidden_dim=768):
        super(U_Adapter, self).__init__()
        if use_cdc:
            self.adapter_conv = Conv2d_cd(in_channels=adapter_dim, out_channels=adapter_dim,
                                          kernel_size=3, stride=1, padding=1, theta=theta)
        else:
            self.adapter_conv = nn.Conv2d(in_channels=adapter_dim, out_channels=adapter_dim,
                                          kernel_size=3, stride=1, padding=1)
        self.cross_attention = My_MHA_UC(dim=adapter_dim)
        self.ln_before = nn.LayerNorm(adapter_dim)
        # nn.init.xavier_uniform_(self.adapter_conv.conv.weight)  # CDC xavier初始化
        # nn.init.zeros_(self.adapter_conv.conv.bias)

        self.adapter_down_1 = nn.Linear(hidden_dim, adapter_dim)  # equivalent to 1 * 1 Conv
        self.adapter_down_2 = nn.Linear(hidden_dim, adapter_dim)  # equivalent to 1 * 1 Conv

        self.adapter_up = nn.Linear(adapter_dim, hidden_dim)  # equivalent to 1 * 1 Conv
        nn.init.xavier_uniform_(self.adapter_down_1.weight)
        nn.init.zeros_(self.adapter_down_1.bias)
        nn.init.xavier_uniform_(self.adapter_down_2.weight)
        nn.init.zeros_(self.adapter_down_2.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        self.act = QuickGELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.dim = adapter_dim

    # @torchsnooper.snoop()
    def forward(self, x, x_c, x_u):
        B, N, C = x.shape

        x_down_1 = self.adapter_down_1(x)  # equivalent to 1 * 1 Conv
        x_down_1 = self.act(x_down_1)

        x_down_2 = self.adapter_down_2(x_c)  # equivalent to 1 * 1 Conv
        x_down_2 = self.act(x_down_2)

        x_cross = self.cross_attention(x_down_2, x_down_1, x_down_1, x_u)
        x_down = self.ln_before(x_cross + x_down_1)

        x_patch = x_down[:, 1:(1 + 14 * 14)]
        x_patch = x_patch.reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)

        x_patch = self.adapter_conv(x_patch)

        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)

        x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
        x_cls = self.adapter_conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        return x_up


class MMDG(nn.Module):
    def __init__(self, dropout_ratio=0.3, exp_rate=2.25, num_sample=20, adapter_dim=8, r_ssp=0.3):
        super(MMDG, self).__init__()
        self.num_encoders = 12
        dim = 768

        vit_b_16 = torchvision.models.vit_b_16(pretrained=True)
        for p in vit_b_16.parameters():
            p.requires_grad = True

        # extract encoder alone and discard CNN (patchify + linear projection) feature extractor, classifer head
        # Refer Encoder() class in https://pytorch.org/vision/main/_modules/torchvision/models/vision_transformer.html

        self.conv_proj1 = vit_b_16.conv_proj
        for p in self.conv_proj1.parameters():
            p.requires_grad = False

        self.conv_proj2 = copy.deepcopy(self.conv_proj1)
        self.conv_proj3 = copy.deepcopy(self.conv_proj1)

        self.class_token1 = vit_b_16.class_token
        self.class_token2 = copy.deepcopy(self.class_token1)
        self.class_token3 = copy.deepcopy(self.class_token1)

        vit = vit_b_16.encoder

        # self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        seq_length = 14 * 14 + 1

        self.pos_embedding1 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))
        self.pos_embedding2 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))
        self.pos_embedding3 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))

        # start building ViT encoder layers
        layer_list = [vit.dropout]
        for i in range(self.num_encoders):
            if i < (self.num_encoders - 1):
                for p in vit.layers[i].parameters():
                    p.requires_grad = False
            layer_list.append(vit.layers[i])
        # add final encoder layer norm
        layer_list.append(vit.ln)

        # assign models for forward pass
        self.ViT_Encoder = nn.Sequential(*layer_list)
        self.fc_all = nn.Sequential(
            nn.Linear(dim + dim + dim, 2),
        )

        self.adapter_1_2_2 = []
        self.adapter_2_1_2 = []
        self.adapter_1_3_2 = []
        self.adapter_3_1_2 = []
        self.adapter_2_3_2 = []
        self.adapter_3_2_2 = []

        for i in range(self.num_encoders):
            self.adapter_1_2_2.append(U_Adapter(adapter_dim=adapter_dim))
            self.adapter_2_1_2.append(U_Adapter(adapter_dim=adapter_dim))
            self.adapter_1_3_2.append(U_Adapter(adapter_dim=adapter_dim))
            self.adapter_3_1_2.append(U_Adapter(adapter_dim=adapter_dim))

        self.adapter_1_2_2 = nn.Sequential(*self.adapter_1_2_2)
        self.adapter_2_1_2 = nn.Sequential(*self.adapter_2_1_2)
        self.adapter_1_3_2 = nn.Sequential(*self.adapter_1_3_2)
        self.adapter_3_1_2 = nn.Sequential(*self.adapter_3_1_2)

        self.mc_drop_list_1 = [MCDropout(p=dropout_ratio) for i in range(self.num_encoders)]
        self.mc_drop_list_2 = [MCDropout(p=dropout_ratio) for i in range(self.num_encoders)]
        self.mc_drop_list_3 = [MCDropout(p=dropout_ratio) for i in range(self.num_encoders)]

        self.exp_rate = exp_rate
        self.num_sample = num_sample
        self.r_ssp = r_ssp

        self.out = None

    # @torchsnooper.snoop()
    def forward(self, x1, x2, x3, domain=None):

        b, c, fh, fw = x1.shape
        x1 = self.conv_proj1(x1)  # b,d,gh,gw
        x1 = x1.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x2 = self.conv_proj2(x2)  # b,d,gh,gw
        x2 = x2.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x3 = self.conv_proj3(x3)  # b,d,gh,gw
        x3 = x3.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x1 = torch.cat((self.class_token1.expand(b, -1, -1), x1), dim=1)  # b,gh*gw+1,d
        x2 = torch.cat((self.class_token2.expand(b, -1, -1), x2), dim=1)  # b,gh*gw+1,d
        x3 = torch.cat((self.class_token3.expand(b, -1, -1), x3), dim=1)  # b,gh*gw+1,d

        proj1 = x1 + self.pos_embedding1
        proj2 = x2 + self.pos_embedding2
        proj3 = x3 + self.pos_embedding3

        proj1 = self.ViT_Encoder[0](proj1)
        proj2 = self.ViT_Encoder[0](proj2)
        proj3 = self.ViT_Encoder[0](proj3)
        for i in range(1, min(len(self.ViT_Encoder), len(self.ViT_Encoder)) - 1):
            x1 = self.ViT_Encoder[i].ln_1(proj1)
            x2 = self.ViT_Encoder[i].ln_1(proj2)
            x3 = self.ViT_Encoder[i].ln_1(proj3)
            x1_c, _ = self.ViT_Encoder[i].self_attention(x1, x1, x1, need_weights=False)
            x2_c, _ = self.ViT_Encoder[i].self_attention(x2, x2, x2, need_weights=False)
            x3_c, _ = self.ViT_Encoder[i].self_attention(x3, x3, x3, need_weights=False)
            x1_c = self.ViT_Encoder[i].dropout(x1_c)
            x2_c = self.ViT_Encoder[i].dropout(x2_c)
            x3_c = self.ViT_Encoder[i].dropout(x3_c)
            x1_uc = cal_uncertainty(x1_c, self.mc_drop_list_1[i - 1], self.exp_rate, self.num_sample)
            x2_uc = cal_uncertainty(x2_c, self.mc_drop_list_2[i - 1], self.exp_rate, self.num_sample)
            x3_uc = cal_uncertainty(x3_c, self.mc_drop_list_3[i - 1], self.exp_rate, self.num_sample)
            x1_a = proj1 + x1_c
            x2_a = proj2 + x2_c
            x3_a = proj3 + x3_c

            y1 = self.ViT_Encoder[i].ln_2(x1_a)
            y2 = self.ViT_Encoder[i].ln_2(x2_a)
            y3 = self.ViT_Encoder[i].ln_2(x3_a)
            y1_m = self.ViT_Encoder[i].mlp(y1)
            y2_m = self.ViT_Encoder[i].mlp(y2)
            y3_m = self.ViT_Encoder[i].mlp(y3)
            proj1 = x1_a + y1_m + self.adapter_1_2_2[i - 1](y1, y2, x2_uc) + self.adapter_1_3_2[i - 1](y1, y3, x3_uc)
            proj2 = x2_a + y2_m + self.adapter_2_1_2[i - 1](y2, y1, x1_uc)
            proj3 = x3_a + y3_m + self.adapter_3_1_2[i - 1](y3, y1, x1_uc)

        proj1 = self.ViT_Encoder[-1](proj1)
        proj2 = self.ViT_Encoder[-1](proj2)
        proj3 = self.ViT_Encoder[-1](proj3)

        logits1 = proj1[:, 0]  # b,d
        logits2 = proj2[:, 0]  # b,d
        logits3 = proj3[:, 0]  # b,d
        logits_all = torch.cat([logits1, logits2, logits3], dim=1)
        out_all = self.fc_all(logits_all)
        self.out = {
            "out_all": out_all,
            "feat_1": logits1,
            "feat_2": logits2,
            "feat_3": logits3,
            "domain": domain,
            "uc_1": x1_uc,
            "uc_2": x2_uc,
            "uc_3": x3_uc,
        }
        return self.out

    def cal_loss(self, spoof_label, loss_func, pmr_loss, prototypes):
        loss_global = loss_func(self.out['out_all'], spoof_label.squeeze(-1))
        if self.out['domain'] is None:
            self.out['domain'] = spoof_label
        loss_1 = pmr_loss(self.out['feat_1'], prototypes[0], self.out['domain'])
        loss_2 = pmr_loss(self.out['feat_2'], prototypes[1], self.out['domain'])
        loss_3 = pmr_loss(self.out['feat_3'], prototypes[2], self.out['domain'])
        # loss = loss_global + self.r_ssp * (loss_1 + loss_2 + loss_3)
        loss_dict = {
            "total": loss_global,
            "m1": self.r_ssp * loss_1,
            "m2": self.r_ssp * loss_2,
            "m3": self.r_ssp * loss_3,
            "uc_1": torch.mean(self.out['uc_1'][:, 0]),
            "uc_2": torch.mean(self.out['uc_2'][:, 0]),
            "uc_3": torch.mean(self.out['uc_3'][:, 0])
        }
        return loss_dict

class Depth_Separable_Convolution(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(Depth_Separable_Convolution, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_chs,
                out_channels=in_chs,
                kernel_size=(3, 3),
                stride=(1, 1),
                groups=in_chs,
                padding=(1, 1),
            ),
            nn.BatchNorm2d(num_features=in_chs),
            nn.ReLU(),
        )
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_chs,
                out_channels=out_chs,
                kernel_size=(1, 1),
                stride=(1, 1),
                groups=1,
                padding=(0, 0),
            ),
            nn.BatchNorm2d(num_features=out_chs),
            nn.ReLU(),
        )
    def forward(self, x):
        x = self.depthwise_conv(x)
        out = self.pointwise_conv(x)
        return out

class MIM(nn.Module):
    def __init__(self, dim=768, patch_size=16, input_resolution=224):
        super(MIM, self).__init__()
        self.W = input_resolution // patch_size
        self.dim = dim
        self.interactive_block = nn.Sequential(
            Depth_Separable_Convolution(self.dim * 2, 256),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.interactive_fc = nn.Linear(self.dim * 2, 256)
        self.mask_generation = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 2*self.dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(2*self.dim),
            nn.Sigmoid()
        )
        self.beta = 1.0
        self.fc1 = nn.Linear(self.dim + 128, self.dim)
        self.fc2 = nn.Linear(self.dim + 128, self.dim)

    def forward(self, m1, m2):
        B, L, D = m1.size()
        W, H = self.W, self.W
        m1_cls_token = m1[:, :1, :]
        m2_cls_token = m2[:, :1, :]
        m1_feature = m1[:, 1:, :]
        m2_feature = m2[:, 1:, :]
        m1_feature = rearrange(m1_feature, "b (h w) d -> b d h w", b=B, d=D, h=H, w=W)
        m2_feature = rearrange(m2_feature, "b (h w) d -> b d h w", b=B, d=D, h=H, w=W)

        pairs = torch.cat([m1_feature, m2_feature], dim=1)
        interactive_feature = self.interactive_block(pairs)

        masks = self.mask_generation(interactive_feature)
        masks = self.beta * masks
        first_mask = masks[:, :self.dim]
        second_mask = masks[:, self.dim:]

        m1_aligned_feature = first_mask * m1_feature
        m1_aligned_feature = rearrange(m1_aligned_feature, "b d h w -> b (h w) d", b=B, d=D, h=H, w=W)
        m1_aligned = torch.cat([m1_cls_token, m1_aligned_feature], dim=1)
        m1_info_token = m1_aligned.mean(1, keepdim=True)  # [B, 1, D]


        m2_aligned_feature = second_mask * m2_feature
        m2_aligned_feature = rearrange(m2_aligned_feature, "b d h w -> b (h w) d", b=B, d=D, h=H, w=W)
        m2_aligned = torch.cat([m2_cls_token, m2_aligned_feature], dim=1)
        m2_info_token = m2_aligned.mean(1, keepdim=True)  # [B, 1, D]

        interactive_cls_token = torch.cat([m1_cls_token, m2_cls_token], dim=-1)
        interactive_cls_token = self.interactive_fc(interactive_cls_token)
        interactive_feature = rearrange(interactive_feature, "b d h w -> b (h w) d", b=B, d=256, h=H, w=W)
        interactive = torch.cat([interactive_cls_token, interactive_feature], dim=1)

        m1_fused = torch.cat([m1_aligned, interactive[:, :, :128]], dim=-1)
        m2_fused = torch.cat([m2_aligned, interactive[:, :, 128:]], dim=-1)
        m1_fused = self.fc1(m1_fused)
        m2_fused = self.fc2(m2_fused)
        return m1_fused, m2_fused, m1_info_token, m2_info_token

class DADM(nn.Module):
    def __init__(self, adapter_dim=8, r_ssp=0.3, missing_modality=False):
        super(DADM, self).__init__()
        self.num_encoders = 12
        dim = 768

        vit_b_16 = torchvision.models.vit_b_16(pretrained=True)
        for p in vit_b_16.parameters():
            p.requires_grad = True

        # extract encoder alone and discard CNN (patchify + linear projection) feature extractor, classifer head
        # Refer Encoder() class in https://pytorch.org/vision/main/_modules/torchvision/models/vision_transformer.html

        self.conv_proj1 = vit_b_16.conv_proj
        for p in self.conv_proj1.parameters():
            p.requires_grad = False

        self.conv_proj2 = copy.deepcopy(self.conv_proj1)
        self.conv_proj3 = copy.deepcopy(self.conv_proj1)

        self.class_token1 = vit_b_16.class_token
        self.class_token2 = copy.deepcopy(self.class_token1)
        self.class_token3 = copy.deepcopy(self.class_token1)

        vit = vit_b_16.encoder

        # self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        seq_length = 14 * 14 + 1

        self.pos_embedding1 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))
        self.pos_embedding2 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))
        self.pos_embedding3 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))

        # start building ViT encoder layers
        layer_list = [vit.dropout]
        for i in range(self.num_encoders):
            if i < (self.num_encoders - 1):
                for p in vit.layers[i].parameters():
                    p.requires_grad = False
            layer_list.append(vit.layers[i])
        # add final encoder layer norm
        layer_list.append(vit.ln)

        # assign models for forward pass
        self.ViT_Encoder = nn.Sequential(*layer_list)
        self.fc_all = nn.Linear(dim + dim + dim, 2, bias=False)
        for p in self.fc_all.parameters():
            p.requires_grad = False
        self.fc_d1 = nn.Linear(dim + dim + dim, 2, bias=False)
        self.fc_d2 = nn.Linear(dim + dim + dim, 2, bias=False)
        self.fc_d3 = nn.Linear(dim + dim + dim, 2, bias=False)

        self.adapter_1 = []
        self.adapter_2 = []
        self.adapter_3 = []
        self.mim_12 = []
        self.mim_13 = []
        self.mim_23 = []

        for i in range(self.num_encoders):
            self.mim_12.append(MIM(dim=dim))
            self.mim_13.append(MIM(dim=dim))
            self.mim_23.append(MIM(dim=dim))
            self.adapter_1.append(CDC_Adapter(adapterdim=adapter_dim))
            self.adapter_2.append(CDC_Adapter(adapterdim=adapter_dim))
            self.adapter_3.append(CDC_Adapter(adapterdim=adapter_dim))

        self.mim_12 = nn.Sequential(*self.mim_12)
        self.mim_13 = nn.Sequential(*self.mim_13)
        self.mim_23 = nn.Sequential(*self.mim_23)
        self.adapter_1 = nn.Sequential(*self.adapter_1)
        self.adapter_2 = nn.Sequential(*self.adapter_2)
        self.adapter_3 = nn.Sequential(*self.adapter_3)

        self.prior_m2 = nn.Parameter(torch.zeros(1, 3, 224, 224))
        self.prior_m3 = nn.Parameter(torch.zeros(1, 3, 224, 224))
        self.r_ssp = r_ssp
        self.missing_modality = missing_modality

        self.out = None

    # @torchsnooper.snoop()
    def forward(self, x1, x2, x3, domain=None, modality_domain=None):
        # if self.missing_modality:
        #     idx = torch.all(x2 == 0., dim=(1, 2, 3))
        #     x2[idx] = self.prior_m2
        #     idx = torch.all(x3 == 0., dim=(1, 2, 3))
        #     x3[idx] = self.prior_m3

        b, c, fh, fw = x1.shape
        x1 = self.conv_proj1(x1)  # b,d,gh,gw
        x1 = x1.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x2 = self.conv_proj2(x2)  # b,d,gh,gw
        x2 = x2.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x3 = self.conv_proj3(x3)  # b,d,gh,gw
        x3 = x3.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x1 = torch.cat((self.class_token1.expand(b, -1, -1), x1), dim=1)  # b,gh*gw+1,d
        x2 = torch.cat((self.class_token2.expand(b, -1, -1), x2), dim=1)  # b,gh*gw+1,d
        x3 = torch.cat((self.class_token3.expand(b, -1, -1), x3), dim=1)  # b,gh*gw+1,d

        proj1 = x1 + self.pos_embedding1
        proj2 = x2 + self.pos_embedding2
        proj3 = x3 + self.pos_embedding3

        proj1 = self.ViT_Encoder[0](proj1)
        proj2 = self.ViT_Encoder[0](proj2)
        proj3 = self.ViT_Encoder[0](proj3)
        MI_loss1 = 0
        MI_loss2 = 0
        MI_loss3 = 0
        for i in range(1, min(len(self.ViT_Encoder), len(self.ViT_Encoder)) - 1):
            x1 = self.ViT_Encoder[i].ln_1(proj1)
            x2 = self.ViT_Encoder[i].ln_1(proj2)
            x3 = self.ViT_Encoder[i].ln_1(proj3)
            x1_c, _ = self.ViT_Encoder[i].self_attention(x1, x1, x1, need_weights=False)
            x2_c, _ = self.ViT_Encoder[i].self_attention(x2, x2, x2, need_weights=False)
            x3_c, _ = self.ViT_Encoder[i].self_attention(x3, x3, x3, need_weights=False)
            x1_c = self.ViT_Encoder[i].dropout(x1_c)
            x2_c = self.ViT_Encoder[i].dropout(x2_c)
            x3_c = self.ViT_Encoder[i].dropout(x3_c)

            x12_aligned, x21_aligned, x12_info, x21_info = self.mim_12[i - 1](x1_c, x2_c)
            x13_aligned, x31_aligned, x13_info, x31_info = self.mim_13[i - 1](x1_c, x3_c)
            x23_aligned, x32_aligned, x23_info, x32_info = self.mim_23[i - 1](x2_c, x3_c)


            # Compute InfoNCE Loss
            MI_loss1 += self.my_mi_loss(x12_info, x21_info)
            MI_loss2 += self.my_mi_loss(x13_info, x31_info)
            MI_loss3 += self.my_mi_loss(x23_info, x32_info)

            x1_aligned = (x12_aligned + x13_aligned) * 0.5
            x2_aligned = (x21_aligned + x23_aligned) * 0.5
            x3_aligned = (x31_aligned + x32_aligned) * 0.5

            if i == 1:
                x1_mi = (x12_info.clone().detach() + x13_info.clone().detach()) * 0.5 / self.num_encoders
                x2_mi = (x21_info.clone().detach() + x23_info.clone().detach()) * 0.5 / self.num_encoders
                x3_mi = (x31_info.clone().detach() + x32_info.clone().detach()) * 0.5 / self.num_encoders
            else:
                x1_mi += (x12_info.clone().detach() + x13_info.clone().detach()) * 0.5 / self.num_encoders
                x2_mi += (x21_info.clone().detach() + x23_info.clone().detach()) * 0.5 / self.num_encoders
                x3_mi += (x31_info.clone().detach() + x32_info.clone().detach()) * 0.5 / self.num_encoders

            x1_a = proj1 + x1_c
            x2_a = proj2 + x2_c
            x3_a = proj3 + x3_c

            y1 = self.ViT_Encoder[i].ln_2(x1_a)
            y2 = self.ViT_Encoder[i].ln_2(x2_a)
            y3 = self.ViT_Encoder[i].ln_2(x3_a)
            y1_m = self.ViT_Encoder[i].mlp(y1)
            y2_m = self.ViT_Encoder[i].mlp(y2)
            y3_m = self.ViT_Encoder[i].mlp(y3)
            proj1 = x1_a + y1_m + self.adapter_1[i - 1](x1_aligned)
            proj2 = x2_a + y2_m + self.adapter_2[i - 1](x2_aligned)
            proj3 = x3_a + y3_m + self.adapter_3[i - 1](x3_aligned)

        proj1 = self.ViT_Encoder[-1](proj1)
        proj2 = self.ViT_Encoder[-1](proj2)
        proj3 = self.ViT_Encoder[-1](proj3)

        logits1 = proj1[:, 0]  # b,d
        logits2 = proj2[:, 0]  # b,d
        logits3 = proj3[:, 0]  # b,d
        logits_all = torch.cat([logits1, logits2, logits3], dim=1)
        out_all = self.fc_all(logits_all)

        if modality_domain is not None:
            idx_d1 = modality_domain.cuda() == 0
            if idx_d1.sum() > 0:
                logits_d1 = logits_all[idx_d1]
                out_d1 = self.fc_d1(logits_d1)
            else:
                out_d1 = None

            idx_d2 = modality_domain.cuda() == 1
            if idx_d2.sum() > 0:
                logits_d2 = logits_all[idx_d2]
                out_d2 = self.fc_d2(logits_d2)
            else:
                out_d2 = None

            idx_d3 = modality_domain.cuda() == 2
            if idx_d3.sum() > 0:
                logits_d3 = logits_all[idx_d3]
                out_d3 = self.fc_d3(logits_d3)
            else:
                out_d3 = None

            self.out = {
                "out_all": out_all,
                "out_d1": out_d1,
                "out_d2": out_d2,
                "out_d3": out_d3,
                "idx_d1": idx_d1,
                "idx_d2": idx_d2,
                "idx_d3": idx_d3,
                "feat_1": logits1,
                "feat_2": logits2,
                "feat_3": logits3,
                "domain": domain,
                "modality_domain": modality_domain,
                "mi_1": x1_mi,
                "mi_2": x2_mi,
                "mi_3": x3_mi,
                "mi_loss1": MI_loss1,
                "mi_loss2": MI_loss2,
                "mi_loss3": MI_loss3,
            }
        else:
            self.out = {
                "out_all": out_all,
                "feat_1": logits1,
                "feat_2": logits2,
                "feat_3": logits3,
                "domain": domain,
                "modality_domain": modality_domain,
                "mi_1": x1_mi,
                "mi_2": x2_mi,
                "mi_3": x3_mi,
                "mi_loss1": MI_loss1,
                "mi_loss2": MI_loss2,
                "mi_loss3": MI_loss3,
            }

        return self.out

    def cal_loss(self, spoof_label, loss_func, pmr_loss, ang_loss, prototypes):
        #loss_global = loss_func(self.out['out_all'], spoof_label.squeeze(-1)) + 0.05 * ang_loss([self.out['feat_1'], self.out['feat_2'], self.out['feat_3']], spoof_label.squeeze(-1), self.out['modality_domain'])
        loss_global = 0.2 * ang_loss([self.out['feat_1'], self.out['feat_2'], self.out['feat_3']], spoof_label.squeeze(-1), self.out['modality_domain'])
        if self.out['out_d1'] is not None:
            loss_global += loss_func(self.out['out_d1'], spoof_label[self.out['idx_d1']].squeeze(-1)) * self.out['idx_d1'].sum() / self.out['idx_d1'].size(0)
        if self.out['out_d2'] is not None:
            loss_global += loss_func(self.out['out_d2'], spoof_label[self.out['idx_d2']].squeeze(-1)) * self.out['idx_d2'].sum() / self.out['idx_d2'].size(0)
        if self.out['out_d3'] is not None:
            loss_global += loss_func(self.out['out_d3'], spoof_label[self.out['idx_d3']].squeeze(-1)) * self.out['idx_d3'].sum() / self.out['idx_d3'].size(0)
        #print(loss_global)
        if self.out['domain'] is None:
            self.out['domain'] = spoof_label
        loss_1 = pmr_loss(self.out['feat_1'], prototypes[0], self.out['domain'])
        loss_2 = pmr_loss(self.out['feat_2'], prototypes[1], self.out['domain'])
        loss_3 = pmr_loss(self.out['feat_3'], prototypes[2], self.out['domain'])
        # loss = loss_global + self.r_ssp * (loss_1 + loss_2 + loss_3)
        loss_dict = {
            "total": loss_global,
            "m1": self.r_ssp * (loss_1 + 0.5 * self.out['mi_loss1']), # or 0. * loss_1
            "m2": self.r_ssp * (loss_2 + 0.5 * self.out['mi_loss2']), # or 0. * loss_2
            "m3": self.r_ssp * (loss_3 + 0.5 * self.out['mi_loss3']), # or 0. * loss_3
            "mi_1": torch.mean(self.out['mi_1'][:, 0]), 
            "mi_2": torch.mean(self.out['mi_2'][:, 0]),
            "mi_3": torch.mean(self.out['mi_3'][:, 0]),
        }
        return loss_dict

    def update_hyperplane(self, alpha=0.995):
        beta0 = normalizer(self.fc_d1.weight.data.squeeze())
        beta1 = normalizer(self.fc_d2.weight.data.squeeze())
        beta2 = normalizer(self.fc_d3.weight.data.squeeze())

        beta0_fr = min([((beta0 * beta1).sum().item(), 0, beta1), ((beta0 * beta2).sum().item(), 1, beta2)])[2]
        beta1_fr = min([((beta1 * beta0).sum().item(), 0, beta0), ((beta1 * beta2).sum().item(), 1, beta2)])[2]
        beta2_fr = min([((beta2 * beta0).sum().item(), 0, beta0), ((beta2 * beta1).sum().item(), 1, beta1)])[2]

        self.fc_d1.weight.data = (normalizer(alpha * beta0 + (1 - alpha) * beta0_fr) * torch.norm(self.fc_d1.weight.data, p=2, dim=-1, keepdim=True))
        self.fc_d2.weight.data = (normalizer(alpha * beta1 + (1 - alpha) * beta1_fr) * torch.norm(self.fc_d2.weight.data, p=2, dim=-1, keepdim=True))
        self.fc_d3.weight.data = (normalizer(alpha * beta2 + (1 - alpha) * beta2_fr) * torch.norm(self.fc_d3.weight.data, p=2, dim=-1, keepdim=True))

        self.fc_all.weight.data = (self.fc_d1.weight.data + self.fc_d2.weight.data + self.fc_d3.weight.data) / 3
        return ((beta0 * beta1).sum() + (beta2 * beta1).sum() + (beta0 * beta2).sum()) / 3

    def InfoNCE_loss(self, m1, m2, temperature=0.07):
        m1 = m1.squeeze()
        m2 = m2.squeeze()
        sim_matrix = F.cosine_similarity(m1.unsqueeze(1), m2.unsqueeze(0), dim=-1) / temperature
        labels = torch.arange(m1.size(0)).to(m1.device)
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

    def my_mi_loss(self, m1, m2):
        mi_estimation = torch.mean((m1 + m2) * 0.5) - \
                         torch.log(torch.mean(torch.exp((m1[torch.randperm(m1.size(0))] + m2[torch.randperm(m2.size(0))]) * 0.5)))
        return -mi_estimation


class MINE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MINE, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, y):
        h = torch.cat((x, y), dim=-1)
        h = F.relu(self.fc1(h))
        return self.fc2(h)


def mine_loss(model, x, y):
    joint = model(x, y)
    marginal = model(x[torch.randperm(x.size(0))], y[torch.randperm(y.size(0))])

    mi_estimation = torch.mean(joint) - torch.log(torch.mean(torch.exp(marginal)))
    return -mi_estimation


