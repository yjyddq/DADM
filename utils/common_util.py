import torch
import math
from torch.optim.lr_scheduler import LRScheduler


class CosineAnnealingLR_with_Restart(LRScheduler):
    """Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::

        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))

    When last_epoch=-1, sets initial lr as lr.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. The original pytorch
    implementation only implements the cosine annealing part of SGDR,
    I added my own implementation of the restarts part.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        T_mult (float): Increase T_max by a factor of T_mult
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_max, T_mult, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.T_mult = T_mult
        self.Te = self.T_max
        self.eta_min = eta_min
        self.current_epoch = last_epoch
        self.lr_history = []

        super(CosineAnnealingLR_with_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        new_lrs = [self.eta_min + (base_lr - self.eta_min) *
                   (1 + math.cos(math.pi * self.current_epoch / self.Te)) / 2
                   for base_lr in self.base_lrs]

        self.lr_history.append(new_lrs)
        return new_lrs

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.current_epoch += 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        ## restart
        if self.current_epoch == self.Te:
            print("restart at epoch {:03d}".format(self.last_epoch + 1))

            ## reset epochs since the last reset
            self.current_epoch = 0

            ## reset the next goal
            self.Te = int(self.Te * self.T_mult)
            self.T_max = self.T_max + self.Te


def forward_model(model, x_rgb, x_d, x_ir, modality):
    if modality == 'RGB':
        return model(x_rgb)
    if modality == 'D':
        return model(x_d)
    # if modality == 'IR':
    #     return model(x_ir)
    if modality == 'RGBD':
        return model(x_rgb, x_d)
    # if modality == 'RGBIR':
    #     return model(x_rgb)
    if modality == 'RGBIR':
        return model(x_rgb, x_ir)
    if modality == 'RGBDIR':
        # print(x_rgb, x_d, x_ir)
        # print(x_rgb.shape, x_d.shape, x_ir.shape)
        return model(x_rgb, x_d, x_ir)

def forward_clip_model(model, x_rgb, x_d, x_ir, modality, mode='train', missing=[]):
    if mode == 'train':
        if modality == 'RGB':
            return model(x_rgb)
        if modality == 'D':
            return model(x_d)
        if modality == 'IR':
            return model(x_ir)
        if modality == 'RGBD':
            return model(x_rgb, x_d)
        if modality == 'DIR':
            return model(x_d, x_ir)
        if modality == 'RGBIR':
            return model(x_rgb, x_ir)
        if modality == 'RGBDIR':
            return model(x_rgb, x_d, x_ir, missing)
    elif mode == 'test':
        if modality == 'RGB':
            return model.forward_test(x_rgb)
        if modality == 'D':
            return model.forward_test(x_d)
        if modality == 'IR':
            return model.forward_test(x_ir)
        if modality == 'RGBD':
            return model.forward_test(x_rgb, x_d)
        if modality == 'DIR':
            return model.forward_test(x_d, x_ir)
        if modality == 'RGBIR':
            return model.forward_test(x_rgb, x_ir)
        if modality == 'RGBDIR':
            return model.forward_test(x_rgb, x_d, x_ir, missing)


def forward_model_with_domain(model, x_rgb, x_d, x_ir, modality, domain, modality_domain):
    if modality == 'RGB':
        return model(x_rgb, domain, modality_domain)
    if modality == 'D':
        return model(x_d, domain, modality_domain)
    if modality == 'RGBD':
        return model(x_rgb, x_d, domain, modality_domain)
    if modality == 'RGBIR':
        return model(x_rgb, x_ir, domain, modality_domain)
    if modality == 'RGBDIR':
        return model(x_rgb, x_d, x_ir, domain, modality_domain)
