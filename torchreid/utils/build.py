# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import torch


def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.lr
        weight_decay = cfg.weight_decay
        if "bias" in key:
            lr = cfg.lr * cfg.bias_lr_factor
            weight_decay = cfg.weight_decay
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.optim == 'sgd':
        optimizer = getattr(torch.optim, cfg.optim)(params, momentum=cfg.momentum)
    else:
        optimizer = getattr(torch.optim, 'Adam')(params)
    return optimizer


def make_optimizer_with_center(cfg, model, center_criterion):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.optim == 'SGD':
        optimizer = getattr(torch.optim, cfg.optim)(params, momentum=cfg.SOLVER.MOMENTUM)
    else:
        optimizer = getattr(torch.optim, cfg.optim)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)
    return optimizer, optimizer_center
