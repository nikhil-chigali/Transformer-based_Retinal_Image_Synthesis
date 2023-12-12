import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import functools


def get_norm_layer(norm_type="instance"):
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def get_scheduler(optimizer, config):
    if config.lr_policy == "lambda":

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + config.epoch_count - config.niter) / float(
                config.niter_decay + 1
            )
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif config.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=config.lr_decay_iters, gamma=0.1
        )
    elif config.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, threshold=0.01, patience=5
        )
    else:
        return NotImplementedError(
            "learning rate policy [%s] is not implemented", config.lr_policy
        )
    return scheduler
