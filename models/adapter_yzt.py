import torch
from torch import nn
# import timm
import math
from torch.nn import functional as F
import pdb


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


class Convpass(nn.Module):
    def __init__(self, adapterdim=8, theta=0.7):
        super(Convpass, self).__init__()

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


# Here we consider shared for easy case. We can also consider unshared adapter_conv1, adapter_conv2, adapter_conv3 later.
class Convpass_3modality_independent(nn.Module):
    def __init__(self, adapterdim=8, theta=0.0):
        super(Convpass_3modality_independent, self).__init__()

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

        x_patch1 = x_down[:, 1:(1 + 14 * 14)].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch2 = x_down[:, (1 + 14 * 14):(1 + 14 * 14 * 2)].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch3 = x_down[:, (1 + 14 * 14 * 2):].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)

        x_patch1 = self.adapter_conv(x_patch1)
        x_patch1 = x_patch1.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)
        x_patch2 = self.adapter_conv(x_patch2)
        x_patch2 = x_patch2.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)
        x_patch3 = self.adapter_conv(x_patch3)
        x_patch3 = x_patch3.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)

        x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
        x_cls = self.adapter_conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_down = torch.cat([x_cls, x_patch1, x_patch2, x_patch3], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        return x_up


# Here we consider shared for easy case. We can also consider unshared adapter_conv1, adapter_conv2, adapter_conv3 later.
class Convpass_2modality_independent(nn.Module):
    def __init__(self, adapterdim=8, theta=0.0):
        super(Convpass_2modality_independent, self).__init__()

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

        x_patch1 = x_down[:, 1:(1 + 14 * 14)].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch2 = x_down[:, (1 + 14 * 14):(1 + 14 * 14 * 2)].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)

        x_patch1 = self.adapter_conv(x_patch1)
        x_patch1 = x_patch1.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)
        x_patch2 = self.adapter_conv(x_patch2)
        x_patch2 = x_patch2.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)

        x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
        x_cls = self.adapter_conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_down = torch.cat([x_cls, x_patch1, x_patch2], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        return x_up


# Here we consider shared for easy case. We can also consider unshared adapter_conv1, adapter_conv2, adapter_conv3 later.
class Convpass_3modality_independent_new(nn.Module):
    def __init__(self, adapterdim=64, theta=0.0):
        super(Convpass_3modality_independent_new, self).__init__()

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

    def forward(self, x, modality):
        B, N, C = x.shape

        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        if modality == 1 or modality == 2 or modality == 3:
            # 1 modality
            x_patch1 = x_down[:, 1:].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)

            x_patch1 = self.adapter_conv(x_patch1)
            x_patch1 = x_patch1.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)

            x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
            x_cls = self.adapter_conv(x_cls)
            x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

            x_down = torch.cat([x_cls, x_patch1], dim=1)

        if modality == 4 or modality == 5 or modality == 6:
            # 2 modality
            x_patch1 = x_down[:, 1:(1 + 14 * 14)].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
            x_patch2 = x_down[:, (1 + 14 * 14):].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)

            x_patch1 = self.adapter_conv(x_patch1)
            x_patch1 = x_patch1.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)
            x_patch2 = self.adapter_conv(x_patch2)
            x_patch2 = x_patch2.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)

            x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
            x_cls = self.adapter_conv(x_cls)
            x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

            x_down = torch.cat([x_cls, x_patch1, x_patch2], dim=1)

        if modality == 7:
            # 3 modality
            x_patch1 = x_down[:, 1:(1 + 14 * 14)].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
            x_patch2 = x_down[:, (1 + 14 * 14):(1 + 14 * 14 * 2)].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
            x_patch3 = x_down[:, (1 + 14 * 14 * 2):].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)

            x_patch1 = self.adapter_conv(x_patch1)
            x_patch1 = x_patch1.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)
            x_patch2 = self.adapter_conv(x_patch2)
            x_patch2 = x_patch2.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)
            x_patch3 = self.adapter_conv(x_patch3)
            x_patch3 = x_patch3.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)

            x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
            x_cls = self.adapter_conv(x_cls)
            x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

            x_down = torch.cat([x_cls, x_patch1, x_patch2, x_patch3], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        return x_up


class Convpass_3modality_independent_AllAdded(nn.Module):
    def __init__(self, adapterdim=8, theta=0.0):
        super(Convpass_3modality_independent_AllAdded, self).__init__()

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

        x_patch1 = x_down[:, 1:(1 + 14 * 14)].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch2 = x_down[:, (1 + 14 * 14):(1 + 14 * 14 * 2)].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch3 = x_down[:, (1 + 14 * 14 * 2):].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)

        x_patch1 = self.adapter_conv(x_patch1)
        x_patch1 = x_patch1.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)
        x_patch2 = self.adapter_conv(x_patch2)
        x_patch2 = x_patch2.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)
        x_patch3 = self.adapter_conv(x_patch3)
        x_patch3 = x_patch3.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)

        x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
        x_cls = self.adapter_conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_down = torch.cat([x_cls, x_patch1, x_patch2, x_patch3], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        x_up[:, 1:(1 + 14 * 14)] = x_up[:, 1:(1 + 14 * 14)] + x_up[:, (1 + 14 * 14):(1 + 14 * 14 * 2)] + x_up[:, (
                                                                                                                             1 + 14 * 14 * 2):]
        x_up[:, (1 + 14 * 14):(1 + 14 * 14 * 2)] = x_up[:, 1:(1 + 14 * 14)]
        x_up[:, (1 + 14 * 14 * 2):] = x_up[:, 1:(1 + 14 * 14)]

        return x_up


# concat all 3 modality for convolution, the convolutional adatper output is shared for all modalities
class Convpass_3modality_ConcatAllConv(nn.Module):
    def __init__(self, adapterdim=8, theta=0.0):
        super(Convpass_3modality_ConcatAllConv, self).__init__()

        self.adapter_conv = Conv2d_cd(in_channels=adapterdim * 3, out_channels=adapterdim, kernel_size=3, stride=1,
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
        ## N = 1 + 196*3
        B, N, C = x.shape

        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        # x_patch = x_down[:, 1:].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch1 = x_down[:, 1:(1 + 14 * 14)].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch2 = x_down[:, (1 + 14 * 14):(1 + 14 * 14 * 2)].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch3 = x_down[:, (1 + 14 * 14 * 2):].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch_3modal = torch.cat([x_patch1, x_patch2, x_patch3], dim=1)

        # x_patch = self.adapter_conv(x_patch)
        x_patch = self.adapter_conv(x_patch_3modal)

        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)
        x_patch = torch.cat([x_patch, x_patch, x_patch], dim=1)
        # x_patch = x_patch.reshape(B, 14 * 14, self.dim, 3).permute(0, 1, 3, 2).reshape(B, 14 * 14 * 3, self.dim)

        x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
        # pdb.set_trace()
        x_cls = torch.cat([x_cls, x_cls, x_cls], dim=1)
        x_cls = self.adapter_conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        return x_up


# concat all 2 modality for convolution, the convolutional adatper output is shared for all 2 modalities
class Convpass_2modality_ConcatAllConv(nn.Module):
    def __init__(self, adapterdim=64, theta=0.0):
        super(Convpass_2modality_ConcatAllConv, self).__init__()

        self.adapter_conv = Conv2d_cd(in_channels=adapterdim * 2, out_channels=adapterdim, kernel_size=3, stride=1,
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
        ## N = 1 + 196*3
        B, N, C = x.shape

        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        # x_patch = x_down[:, 1:].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch1 = x_down[:, 1:(1 + 14 * 14)].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch2 = x_down[:, (1 + 14 * 14):(1 + 14 * 14 * 2)].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch3 = x_down[:, (1 + 14 * 14 * 2):].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)

        x_patch12_2modal = torch.cat([x_patch1, x_patch2], dim=1)
        x_patch13_2modal = torch.cat([x_patch1, x_patch3], dim=1)
        x_patch23_2modal = torch.cat([x_patch2, x_patch3], dim=1)

        # x_patch = self.adapter_conv(x_patch)
        x_patch12 = self.adapter_conv(x_patch12_2modal)
        x_patch13 = self.adapter_conv(x_patch13_2modal)
        x_patch23 = self.adapter_conv(x_patch23_2modal)

        x_patch12 = x_patch12.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)
        x_patch13 = x_patch13.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)
        x_patch23 = x_patch23.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)

        x_patch = torch.cat(
            [(x_patch12 + x_patch13) / 2.0, (x_patch12 + x_patch23) / 2.0, (x_patch13 + x_patch23) / 2.0], dim=1)
        # x_patch = torch.cat([x_patch, x_patch, x_patch], dim=1)

        x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
        # pdb.set_trace()
        x_cls = torch.cat([x_cls, x_cls], dim=1)
        x_cls = self.adapter_conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        return x_up


# concat all 2 modality for convolution, the convolutional adatper output is shared for all 2 modalities
class Convpass_Concat2ModalConv_Only2modal(nn.Module):
    def __init__(self, adapterdim=64, theta=0.0):
        super(Convpass_Concat2ModalConv_Only2modal, self).__init__()

        self.adapter_conv = Conv2d_cd(in_channels=adapterdim * 2, out_channels=adapterdim, kernel_size=3, stride=1,
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
        ## N = 1 + 196*3
        B, N, C = x.shape

        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        # x_patch = x_down[:, 1:].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch1 = x_down[:, 1:(1 + 14 * 14)].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch2 = x_down[:, (1 + 14 * 14):(1 + 14 * 14 * 2)].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)

        x_patch12_2modal = torch.cat([x_patch1, x_patch2], dim=1)

        # x_patch = self.adapter_conv(x_patch)
        x_patch12 = self.adapter_conv(x_patch12_2modal)

        x_patch12 = x_patch12.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)

        x_patch = torch.cat([x_patch12, x_patch12], dim=1)
        # x_patch = torch.cat([x_patch, x_patch, x_patch], dim=1)

        x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
        # pdb.set_trace()
        x_cls = torch.cat([x_cls, x_cls], dim=1)
        x_cls = self.adapter_conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        return x_up


# concat all 2 modality for convolution, the convolutional adatper output is shared for all 2 modalities
class Convpass_2modality_ConcatAllConv_new(nn.Module):
    def __init__(self, adapterdim=64, theta=0.0):
        super(Convpass_2modality_ConcatAllConv_new, self).__init__()

        self.adapter_conv = Conv2d_cd(in_channels=adapterdim * 2, out_channels=adapterdim, kernel_size=3, stride=1,
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
        ## N = 1 + 196*3
        B, N, C = x.shape

        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        # x_patch = x_down[:, 1:].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch1 = x_down[:, 1:(1 + 14 * 14)].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch2 = x_down[:, (1 + 14 * 14):].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)

        x_patch12_2modal = torch.cat([x_patch1, x_patch2], dim=1)

        # x_patch = self.adapter_conv(x_patch)
        x_patch12 = self.adapter_conv(x_patch12_2modal)

        x_patch12 = x_patch12.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)

        x_patch = torch.cat([x_patch12, x_patch12], dim=1)
        # x_patch = torch.cat([x_patch, x_patch, x_patch], dim=1)

        x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
        # pdb.set_trace()
        x_cls = torch.cat([x_cls, x_cls], dim=1)
        x_cls = self.adapter_conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        return x_up


# concat all 3 modality for convolution, the convolutional adatper output is shared for all modalities
class Convpass_3modality_ConcatAllConv_ModalityRefine(nn.Module):
    def __init__(self, adapterdim=8, theta=0.0):
        super(Convpass_3modality_ConcatAllConv_ModalityRefine, self).__init__()

        self.adapter_conv = Conv2d_cd(in_channels=adapterdim * 3, out_channels=adapterdim, kernel_size=3, stride=1,
                                      padding=1, theta=theta)

        # nn.init.xavier_uniform_(self.adapter_conv.weight)

        self.adapter_down = nn.Linear(768, adapterdim)  # equivalent to 1 * 1 Conv
        self.adapter_up = nn.Linear(adapterdim, 768)  # equivalent to 1 * 1 Conv
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        self.CAconv = nn.Conv2d(adapterdim * 3, 3, kernel_size=1, padding=0, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = adapterdim

    def forward(self, x):
        ## N = 1 + 196*3
        B, N, C = x.shape

        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        # x_patch = x_down[:, 1:].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch1 = x_down[:, 1:(1 + 14 * 14)].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch2 = x_down[:, (1 + 14 * 14):(1 + 14 * 14 * 2)].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch3 = x_down[:, (1 + 14 * 14 * 2):].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch_3modal = torch.cat([x_patch1, x_patch2, x_patch3], dim=1)

        # x_patch = self.adapter_conv(x_patch)
        x_patch = self.adapter_conv(x_patch_3modal)
        ChannelAtten = self.sigmoid(self.CAconv(self.avgpool(x_patch_3modal)).squeeze(-1).squeeze(-1))

        # pdb.set_trace()

        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)
        x_patch = torch.cat([ChannelAtten[:, 0].unsqueeze(-1).unsqueeze(-1) * x_patch,
                             ChannelAtten[:, 1].unsqueeze(-1).unsqueeze(-1) * x_patch,
                             ChannelAtten[:, 2].unsqueeze(-1).unsqueeze(-1) * x_patch], dim=1)
        # x_patch = x_patch.reshape(B, 14 * 14, self.dim, 3).permute(0, 1, 3, 2).reshape(B, 14 * 14 * 3, self.dim)

        x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
        # pdb.set_trace()
        x_cls = torch.cat([x_cls, x_cls, x_cls], dim=1)
        x_cls = self.adapter_conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        return x_up


class Convpass_2modality_ConcatAllConv_ModalityRefine(nn.Module):
    def __init__(self, adapterdim=64, theta=0.0):
        super(Convpass_2modality_ConcatAllConv_ModalityRefine, self).__init__()

        self.adapter_conv = Conv2d_cd(in_channels=adapterdim * 2, out_channels=adapterdim, kernel_size=3, stride=1,
                                      padding=1, theta=theta)

        # nn.init.xavier_uniform_(self.adapter_conv.weight)

        self.adapter_down = nn.Linear(768, adapterdim)  # equivalent to 1 * 1 Conv
        self.adapter_up = nn.Linear(adapterdim, 768)  # equivalent to 1 * 1 Conv
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        self.CAconv = nn.Conv2d(adapterdim * 2, 2, kernel_size=1, padding=0, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = adapterdim

    def forward(self, x):
        ## N = 1 + 196*3
        B, N, C = x.shape

        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        # x_patch = x_down[:, 1:].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch1 = x_down[:, 1:(1 + 14 * 14)].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch2 = x_down[:, (1 + 14 * 14):(1 + 14 * 14 * 2)].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch3 = x_down[:, (1 + 14 * 14 * 2):].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)

        x_patch12_2modal = torch.cat([x_patch1, x_patch2], dim=1)
        x_patch13_2modal = torch.cat([x_patch1, x_patch3], dim=1)
        x_patch23_2modal = torch.cat([x_patch2, x_patch3], dim=1)

        # x_patch = self.adapter_conv(x_patch)
        x_patch12 = self.adapter_conv(x_patch12_2modal)
        ChannelAtten12 = self.sigmoid(self.CAconv(self.avgpool(x_patch12_2modal)).squeeze(-1).squeeze(-1))
        x_patch13 = self.adapter_conv(x_patch13_2modal)
        ChannelAtten13 = self.sigmoid(self.CAconv(self.avgpool(x_patch13_2modal)).squeeze(-1).squeeze(-1))
        x_patch23 = self.adapter_conv(x_patch23_2modal)
        ChannelAtten23 = self.sigmoid(self.CAconv(self.avgpool(x_patch23_2modal)).squeeze(-1).squeeze(-1))

        x_patch12 = x_patch12.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)
        x_patch13 = x_patch13.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)
        x_patch23 = x_patch23.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)

        x_patch = torch.cat([(ChannelAtten12[:, 0].unsqueeze(-1).unsqueeze(-1) * x_patch12 + ChannelAtten13[:,
                                                                                             0].unsqueeze(-1).unsqueeze(
            -1) * x_patch13) / 2.0, (ChannelAtten12[:, 1].unsqueeze(-1).unsqueeze(-1) * x_patch12 + ChannelAtten23[:,
                                                                                                    0].unsqueeze(
            -1).unsqueeze(-1) * x_patch23) / 2.0, (
                                         ChannelAtten13[:, 1].unsqueeze(-1).unsqueeze(-1) * x_patch13 + ChannelAtten23[
                                                                                                        :, 1].unsqueeze(
                                     -1).unsqueeze(-1) * x_patch23) / 2.0], dim=1)

        x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
        # pdb.set_trace()
        x_cls = torch.cat([x_cls, x_cls], dim=1)
        x_cls = self.adapter_conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        return x_up


class Convpass_Concat2ModalConv_ModalityRefine_Only2modal(nn.Module):
    def __init__(self, adapterdim=64, theta=0.0):
        super(Convpass_Concat2ModalConv_ModalityRefine_Only2modal, self).__init__()

        self.adapter_conv = Conv2d_cd(in_channels=adapterdim * 2, out_channels=adapterdim, kernel_size=3, stride=1,
                                      padding=1, theta=theta)

        # nn.init.xavier_uniform_(self.adapter_conv.weight)

        self.adapter_down = nn.Linear(768, adapterdim)  # equivalent to 1 * 1 Conv
        self.adapter_up = nn.Linear(adapterdim, 768)  # equivalent to 1 * 1 Conv
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        self.CAconv = nn.Conv2d(adapterdim * 2, 2, kernel_size=1, padding=0, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = adapterdim

    def forward(self, x):
        ## N = 1 + 196*3
        B, N, C = x.shape

        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        # x_patch = x_down[:, 1:].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch1 = x_down[:, 1:(1 + 14 * 14)].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch2 = x_down[:, (1 + 14 * 14):(1 + 14 * 14 * 2)].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)

        x_patch12_2modal = torch.cat([x_patch1, x_patch2], dim=1)

        # x_patch = self.adapter_conv(x_patch)
        x_patch12 = self.adapter_conv(x_patch12_2modal)
        ChannelAtten12 = self.sigmoid(self.CAconv(self.avgpool(x_patch12_2modal)).squeeze(-1).squeeze(-1))

        x_patch12 = x_patch12.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)

        x_patch = torch.cat([ChannelAtten12[:, 0].unsqueeze(-1).unsqueeze(-1) * x_patch12,
                             ChannelAtten12[:, 1].unsqueeze(-1).unsqueeze(-1) * x_patch12], dim=1)

        x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
        # pdb.set_trace()
        x_cls = torch.cat([x_cls, x_cls], dim=1)
        x_cls = self.adapter_conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        return x_up


# Only two modalities
class Convpass_2modality_ConcatAllConv_ModalityRefine_new(nn.Module):
    def __init__(self, adapterdim=64, theta=0.0):
        super(Convpass_2modality_ConcatAllConv_ModalityRefine_new, self).__init__()

        self.adapter_conv = Conv2d_cd(in_channels=adapterdim * 2, out_channels=adapterdim, kernel_size=3, stride=1,
                                      padding=1, theta=theta)

        # nn.init.xavier_uniform_(self.adapter_conv.weight)

        self.adapter_down = nn.Linear(768, adapterdim)  # equivalent to 1 * 1 Conv
        self.adapter_up = nn.Linear(adapterdim, 768)  # equivalent to 1 * 1 Conv
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        self.CAconv = nn.Conv2d(adapterdim * 2, 2, kernel_size=1, padding=0, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = adapterdim

    def forward(self, x):
        ## N = 1 + 196*3
        B, N, C = x.shape

        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        # x_patch = x_down[:, 1:].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch1 = x_down[:, 1:(1 + 14 * 14)].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch2 = x_down[:, (1 + 14 * 14):].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)

        x_patch12_2modal = torch.cat([x_patch1, x_patch2], dim=1)

        # x_patch = self.adapter_conv(x_patch)
        x_patch12 = self.adapter_conv(x_patch12_2modal)
        ChannelAtten12 = self.sigmoid(self.CAconv(self.avgpool(x_patch12_2modal)).squeeze(-1).squeeze(-1))

        x_patch12 = x_patch12.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)

        x_patch = torch.cat([ChannelAtten12[:, 0].unsqueeze(-1).unsqueeze(-1) * x_patch12,
                             ChannelAtten12[:, 1].unsqueeze(-1).unsqueeze(-1) * x_patch12], dim=1)

        x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
        # pdb.set_trace()
        x_cls = torch.cat([x_cls, x_cls], dim=1)
        x_cls = self.adapter_conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        return x_up


# concat all 3 modality for convolution, the convolutional adatper output is shared for all modalities
class Convpass_3modality_ConcatAllConv_noClassToken(nn.Module):
    def __init__(self, adapterdim=8, theta=0.0):
        super(Convpass_3modality_ConcatAllConv_noClassToken, self).__init__()

        self.adapter_conv = Conv2d_cd(in_channels=adapterdim * 3, out_channels=adapterdim, kernel_size=3, stride=1,
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
        ## N = 1 + 196*3
        B, N, C = x.shape

        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        # x_patch = x_down[:, 1:].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch1 = x_down[:, 1:(1 + 14 * 14)].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch2 = x_down[:, (1 + 14 * 14):(1 + 14 * 14 * 2)].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch3 = x_down[:, (1 + 14 * 14 * 2):].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch_3modal = torch.cat([x_patch1, x_patch2, x_patch3], dim=1)

        # x_patch = self.adapter_conv(x_patch)
        x_patch = self.adapter_conv(x_patch_3modal)

        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)
        x_patch = torch.cat([x_patch, x_patch, x_patch], dim=1)
        # x_patch = x_patch.reshape(B, 14 * 14, self.dim, 3).permute(0, 1, 3, 2).reshape(B, 14 * 14 * 3, self.dim)

        x_cls = x_down[:, :1]

        x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        return x_up


# concat all 3 modality for convolution, the convolutional adatper output is shared for all modalities
class Convpass_3modality_ConcatAllConv3output_noClassToken(nn.Module):
    def __init__(self, adapterdim=8, theta=0.0):
        super(Convpass_3modality_ConcatAllConv3output_noClassToken, self).__init__()

        self.adapter_conv = Conv2d_cd(in_channels=adapterdim * 3, out_channels=adapterdim * 3, kernel_size=3, stride=1,
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
        ## N = 1 + 196*3
        B, N, C = x.shape

        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        # x_patch = x_down[:, 1:].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch1 = x_down[:, 1:(1 + 14 * 14)].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch2 = x_down[:, (1 + 14 * 14):(1 + 14 * 14 * 2)].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch3 = x_down[:, (1 + 14 * 14 * 2):].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch_3modal = torch.cat([x_patch1, x_patch2, x_patch3], dim=1)

        # x_patch = self.adapter_conv(x_patch)
        x_patch = self.adapter_conv(x_patch_3modal)

        x_patch = x_patch.reshape(B, 14 * 14, self.dim, 3).permute(0, 1, 3, 2).reshape(B, 14 * 14 * 3, self.dim)

        x_cls = x_down[:, :1]

        x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        return x_up


class vit_CDCadapter(nn.Module):
    def __init__(self, transformer_encoder, adapterdim=8, theta=0.7):
        super(vit_CDCadapter, self).__init__()

        self.ln1 = transformer_encoder.ln_1
        # self.ln1 = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)

        self.attention = transformer_encoder.self_attention
        self.drop = transformer_encoder.dropout

        # conv layer in adapters
        self.mhsa_conv_pass = Convpass(adapterdim=adapterdim, theta=theta)
        self.ffn_conv_pass = Convpass(adapterdim=adapterdim, theta=theta)

        # Feed Forward Layers
        self.ln_2 = transformer_encoder.ln_2
        # self.ln_2 = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)

        self.mlp = transformer_encoder.mlp

    def forward(self, x):
        norm_x = self.ln1(x)
        conv_pass = self.mhsa_conv_pass(norm_x)
        attention, _ = self.attention(query=norm_x, key=norm_x, value=norm_x)
        attention = self.drop(attention)

        mhsa = attention + x + conv_pass

        norm_mhsa = self.ln_2(mhsa)

        ffn = self.mlp(norm_mhsa) + mhsa + self.ffn_conv_pass(norm_mhsa)

        return ffn


class vit_CDCadapter_MHSA(nn.Module):
    def __init__(self, transformer_encoder, adapterdim=8, theta=0.7):
        super(vit_CDCadapter_MHSA, self).__init__()

        self.ln1 = transformer_encoder.ln_1
        # self.ln1 = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)

        self.attention = transformer_encoder.self_attention
        self.drop = transformer_encoder.dropout

        # conv layer in adapters
        self.mhsa_conv_pass = Convpass(adapterdim=adapterdim, theta=theta)

        # Feed Forward Layers
        self.ln_2 = transformer_encoder.ln_2
        # self.ln_2 = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)

        self.mlp = transformer_encoder.mlp

    def forward(self, x):
        norm_x = self.ln1(x)
        conv_pass = self.mhsa_conv_pass(norm_x)
        attention, _ = self.attention(query=norm_x, key=norm_x, value=norm_x)
        attention = self.drop(attention)

        mhsa = attention + x + conv_pass

        norm_mhsa = self.ln_2(mhsa)

        ffn = self.mlp(norm_mhsa) + mhsa

        return ffn


class vit_CDCadapter_FFN(nn.Module):
    def __init__(self, transformer_encoder, adapterdim=8, theta=0.7):
        super(vit_CDCadapter_FFN, self).__init__()

        self.ln1 = transformer_encoder.ln_1
        # self.ln1 = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)

        self.attention = transformer_encoder.self_attention
        self.drop = transformer_encoder.dropout

        self.ffn_conv_pass = Convpass(adapterdim=adapterdim, theta=theta)

        # Feed Forward Layers
        self.ln_2 = transformer_encoder.ln_2
        # self.ln_2 = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)

        self.mlp = transformer_encoder.mlp

    def forward(self, x):
        norm_x = self.ln1(x)
        attention, _ = self.attention(query=norm_x, key=norm_x, value=norm_x)
        attention = self.drop(attention)

        mhsa = attention + x

        norm_mhsa = self.ln_2(mhsa)

        ffn = self.mlp(norm_mhsa) + mhsa + self.ffn_conv_pass(norm_mhsa)

        return ffn


class vit_CDCadapter_3modality(nn.Module):
    def __init__(self, transformer_encoder, adapterdim=8, theta=0.0, adapter=Convpass_3modality_ConcatAllConv):
        super(vit_CDCadapter_3modality, self).__init__()

        self.ln1 = transformer_encoder.ln_1
        # self.ln1 = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)

        self.attention = transformer_encoder.self_attention
        self.drop = transformer_encoder.dropout

        # conv layer in adapters
        self.mhsa_conv_pass = adapter(adapterdim=adapterdim, theta=theta)
        self.ffn_conv_pass = adapter(adapterdim=adapterdim, theta=theta)

        # Feed Forward Layers
        self.ln_2 = transformer_encoder.ln_2
        # self.ln_2 = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)

        self.mlp = transformer_encoder.mlp

    def forward(self, x):
        norm_x = self.ln1(x)
        conv_pass = self.mhsa_conv_pass(norm_x)
        attention, _ = self.attention(query=norm_x, key=norm_x, value=norm_x)
        attention = self.drop(attention)

        mhsa = attention + x + conv_pass

        norm_mhsa = self.ln_2(mhsa)

        ffn = self.mlp(norm_mhsa) + mhsa + self.ffn_conv_pass(norm_mhsa)

        return ffn


class vit_CDCadapter_2modality_3modality_MHSA(nn.Module):
    def __init__(self, transformer_encoder, adapterdim=8, theta=0.0, adapter=Convpass_3modality_ConcatAllConv):
        super(vit_CDCadapter_2modality_3modality_MHSA, self).__init__()

        self.ln1 = transformer_encoder.ln_1
        # self.ln1 = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)

        self.attention = transformer_encoder.self_attention
        self.drop = transformer_encoder.dropout

        # conv layer in adapters
        self.mhsa_conv_pass = adapter(adapterdim=adapterdim, theta=theta)

        # Feed Forward Layers
        self.ln_2 = transformer_encoder.ln_2
        # self.ln_2 = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)

        self.mlp = transformer_encoder.mlp

    def forward(self, x):
        norm_x = self.ln1(x)
        conv_pass = self.mhsa_conv_pass(norm_x)
        attention, _ = self.attention(query=norm_x, key=norm_x, value=norm_x)
        attention = self.drop(attention)

        mhsa = attention + x + conv_pass

        norm_mhsa = self.ln_2(mhsa)

        ffn = self.mlp(norm_mhsa) + mhsa

        return ffn


class vit_CDCadapter_2modality_3modality_FFN(nn.Module):
    def __init__(self, transformer_encoder, adapterdim=8, theta=0.0, adapter=Convpass_3modality_ConcatAllConv):
        super(vit_CDCadapter_2modality_3modality_FFN, self).__init__()

        self.ln1 = transformer_encoder.ln_1
        # self.ln1 = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)

        self.attention = transformer_encoder.self_attention
        self.drop = transformer_encoder.dropout

        # conv layer in adapters
        self.ffn_conv_pass = adapter(adapterdim=adapterdim, theta=theta)

        # Feed Forward Layers
        self.ln_2 = transformer_encoder.ln_2
        # self.ln_2 = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)

        self.mlp = transformer_encoder.mlp

    def forward(self, x):
        norm_x = self.ln1(x)
        attention, _ = self.attention(query=norm_x, key=norm_x, value=norm_x)
        attention = self.drop(attention)

        mhsa = attention + x

        norm_mhsa = self.ln_2(mhsa)

        ffn = self.mlp(norm_mhsa) + mhsa + self.ffn_conv_pass(norm_mhsa)

        return ffn


class vit_CDCadapter_3modality_2adapter(nn.Module):
    def __init__(self, transformer_encoder, adapterdim=64, theta=0.5, adapter1=Convpass_3modality_independent,
                 adapter2=Convpass_3modality_ConcatAllConv_ModalityRefine):
        super(vit_CDCadapter_3modality_2adapter, self).__init__()

        self.ln1 = transformer_encoder.ln_1
        # self.ln1 = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)

        self.attention = transformer_encoder.self_attention
        self.drop = transformer_encoder.dropout

        # conv layer in adapters
        self.mhsa_conv_pass1 = adapter1(adapterdim=adapterdim, theta=0.0)
        self.ffn_conv_pass1 = adapter1(adapterdim=adapterdim, theta=0.0)

        self.mhsa_conv_pass2 = adapter2(adapterdim=adapterdim, theta=0.0)
        self.ffn_conv_pass2 = adapter2(adapterdim=adapterdim, theta=0.0)

        # Feed Forward Layers
        self.ln_2 = transformer_encoder.ln_2
        # self.ln_2 = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)

        self.mlp = transformer_encoder.mlp

        self.theta = theta

    def forward(self, x):
        norm_x = self.ln1(x)
        conv_pass1 = self.mhsa_conv_pass1(norm_x)
        conv_pass2 = self.mhsa_conv_pass2(norm_x)
        attention, _ = self.attention(query=norm_x, key=norm_x, value=norm_x)
        attention = self.drop(attention)

        mhsa = attention + x + self.theta * conv_pass1 + (1 - self.theta) * conv_pass2

        norm_mhsa = self.ln_2(mhsa)

        ffn = self.mlp(norm_mhsa) + mhsa + self.theta * self.ffn_conv_pass1(norm_mhsa) + (
                    1 - self.theta) * self.ffn_conv_pass2(norm_mhsa)

        return ffn


class vit_CDCadapter_3modality_DynamicAdapter(nn.Module):
    def __init__(self, transformer_encoder, adapterdim=64, theta2modality=0.5, theta3modality=0.4,
                 adapter1=Convpass_3modality_independent_new,
                 adapter2=Convpass_2modality_ConcatAllConv_ModalityRefine_new,
                 adapter3=Convpass_3modality_ConcatAllConv_ModalityRefine):
        super(vit_CDCadapter_3modality_DynamicAdapter, self).__init__()

        self.ln1 = transformer_encoder.ln_1
        # self.ln1 = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)

        self.attention = transformer_encoder.self_attention
        self.drop = transformer_encoder.dropout

        # conv layer in adapters
        self.mhsa_conv_pass1 = adapter1(adapterdim=adapterdim, theta=0.0)
        self.ffn_conv_pass1 = adapter1(adapterdim=adapterdim, theta=0.0)

        self.mhsa_conv_pass2 = adapter2(adapterdim=adapterdim, theta=0.0)
        self.ffn_conv_pass2 = adapter2(adapterdim=adapterdim, theta=0.0)

        self.mhsa_conv_pass3 = adapter3(adapterdim=adapterdim, theta=0.0)
        self.ffn_conv_pass3 = adapter3(adapterdim=adapterdim, theta=0.0)

        # Feed Forward Layers
        self.ln_2 = transformer_encoder.ln_2
        # self.ln_2 = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)

        self.mlp = transformer_encoder.mlp

        self.theta2modality = theta2modality
        self.theta3modality = theta3modality

    def forward(self, x, modality):

        norm_x = self.ln1(x)

        # All cases
        conv_pass1 = self.mhsa_conv_pass1(norm_x, modality)

        # Only with two modalities
        if modality == 4 or modality == 5 or modality == 6:
            conv_pass2 = self.mhsa_conv_pass2(norm_x)

        # 3 modality
        if modality == 7:
            conv_pass3 = self.mhsa_conv_pass3(norm_x)

        attention, _ = self.attention(query=norm_x, key=norm_x, value=norm_x)
        attention = self.drop(attention)

        # 1 modality
        if modality == 1 or modality == 2 or modality == 3:
            mhsa = attention + x + conv_pass1
        # 2 modality
        if modality == 4 or modality == 5 or modality == 6:
            mhsa = attention + x + self.theta2modality * conv_pass1 + (1 - self.theta2modality) * conv_pass2
        # 3 modality
        if modality == 7:
            mhsa = attention + x + self.theta3modality * conv_pass1 + (1 - self.theta3modality) * conv_pass3

        norm_mhsa = self.ln_2(mhsa)

        # 1 modality
        if modality == 1 or modality == 2 or modality == 3:
            ffn = self.mlp(norm_mhsa) + mhsa + self.ffn_conv_pass1(norm_mhsa, modality)
        # 2 modality
        if modality == 4 or modality == 5 or modality == 6:
            ffn = self.mlp(norm_mhsa) + mhsa + self.theta2modality * self.ffn_conv_pass1(norm_mhsa, modality) + (
                        1 - self.theta2modality) * self.ffn_conv_pass2(norm_mhsa)
        # 3 modality
        if modality == 7:
            ffn = self.mlp(norm_mhsa) + mhsa + self.theta3modality * self.ffn_conv_pass1(norm_mhsa, modality) + (
                        1 - self.theta3modality) * self.ffn_conv_pass3(norm_mhsa)

        return ffn


class vit_CDCadapter_3modality_3adapter(nn.Module):
    def __init__(self, transformer_encoder, adapterdim=64, theta1=0.33, theta2=0.33, theta3=0.33,
                 adapter1=Convpass_3modality_independent, adapter2=Convpass_3modality_ConcatAllConv_ModalityRefine,
                 adapter3=Convpass_3modality_ConcatAllConv_ModalityRefine):
        super(vit_CDCadapter_3modality_3adapter, self).__init__()

        self.ln1 = transformer_encoder.ln_1
        # self.ln1 = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)

        self.attention = transformer_encoder.self_attention
        self.drop = transformer_encoder.dropout

        # conv layer in adapters
        self.mhsa_conv_pass1 = adapter1(adapterdim=adapterdim, theta=0.0)
        self.ffn_conv_pass1 = adapter1(adapterdim=adapterdim, theta=0.0)

        self.mhsa_conv_pass2 = adapter2(adapterdim=adapterdim, theta=0.0)
        self.ffn_conv_pass2 = adapter2(adapterdim=adapterdim, theta=0.0)

        self.mhsa_conv_pass3 = adapter3(adapterdim=adapterdim, theta=0.0)
        self.ffn_conv_pass3 = adapter3(adapterdim=adapterdim, theta=0.0)

        # Feed Forward Layers
        self.ln_2 = transformer_encoder.ln_2
        # self.ln_2 = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)

        self.mlp = transformer_encoder.mlp

        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3

    def forward(self, x):
        norm_x = self.ln1(x)
        conv_pass1 = self.mhsa_conv_pass1(norm_x)
        conv_pass2 = self.mhsa_conv_pass2(norm_x)
        conv_pass3 = self.mhsa_conv_pass3(norm_x)
        attention, _ = self.attention(query=norm_x, key=norm_x, value=norm_x)
        attention = self.drop(attention)

        mhsa = attention + x + self.theta1 * conv_pass1 + self.theta2 * conv_pass2 + self.theta3 * conv_pass3

        norm_mhsa = self.ln_2(mhsa)

        ffn = self.mlp(norm_mhsa) + mhsa + self.theta1 * self.ffn_conv_pass1(
            norm_mhsa) + self.theta2 * self.ffn_conv_pass2(norm_mhsa) + self.theta3 * self.ffn_conv_pass3(norm_mhsa)

        return ffn

