import os
import random
import numpy as np
import torch
from torch import nn
import torch.distributed as dist


def save_checkpoint_ds(engine, epoch, best_val_loss, best_val_epoch, args):
    client_sd = {}
    client_sd['last_epoch'] = epoch
    client_sd['best_val_loss'] = best_val_loss
    client_sd['best_val_epoch'] = best_val_epoch
    engine.save_checkpoint(args.checkpoint_dir, epoch, client_state=client_sd)


def load_checkpoint_ds(engine, args):
    _, client_sd = engine.load_checkpoint(args.checkpoint_dir, args.load_ckpt_id)
    last_epoch = client_sd['last_epoch']
    best_val_loss = client_sd['best_val_loss']
    best_val_epoch = client_sd['best_val_epoch']
    return engine, last_epoch, best_val_loss, best_val_epoch


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


def fix_random_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def print_main(*args, **kwargs):
    # Check if the current process is the main process (rank 0)
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        # If not in a distributed setup, print normally
        print(*args, **kwargs)


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False
