import os
import tqdm
import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import deepspeed
import subprocess
import argparse
import json
import math
import sys
import copy
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split
from data import DepthToSketchDataset, CondCvtrTransform
from nns.model import CondCvtr
from utils import print_main, fix_random_seeds, save_checkpoint_ds, load_checkpoint_ds, has_batchnorms
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from torch.utils.data.distributed import DistributedSampler
from nns.lr_scheduler import WarmupCosineLR


def get_args_parser():
    parser = argparse.ArgumentParser('Spectrum-OCT', add_help=False)

    # File paths
    parser.add_argument('--config_file', default='cfg/segformer.json', type=str,
        help='Path to the model configuration file.')

    # Checkpoints
    parser.add_argument('--checkpoint_dir', default='runs/segformer', type=str, help='Path to the checkpoint directory.')
    parser.add_argument('--load_ckpt_id', default=-1, type=int, help='Checkpoint ID.')

    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--seed', default=42, type=int, help='Random seed.')
    parser.add_argument('--train_ratio', default=0.9, type=float, help='Ratio of training set to the whole dataset.')
    parser.add_argument("--local_rank", default=0, type=int, help='Used for distributed training, just leave as default.')

    # wandb
    parser.add_argument('--project_name', default='C2C', type=str)
    parser.add_argument('--experiment_name', default='experiment', type=str)
    return parser


def init_dist(args, port=29500):
    # for name, value in os.environ.items():
    #     print("{0}: {1}".format(name, value))
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    args.global_rank = int(os.environ['SLURM_PROCID'])
    args.local_rank = int(os.environ['SLURM_LOCALID'])
    args.world_size = int(os.environ['SLURM_NTASKS'])
    torch.cuda.set_device(args.local_rank)
    addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['RANK'] = str(args.global_rank)
    os.environ['LOCAL_RANK'] = str(args.local_rank)
    port = os.environ.get('PORT', port)
    os.environ['MASTER_PORT'] = str(port)
    print(f"proc_id: {args.global_rank}; local_rank: {args.local_rank}; ntasks: {args.world_size}; node_list: {node_list}; num_gpus: {num_gpus}; addr: {addr}; port: {port}")
    deepspeed.init_distributed()
    return args


def train(args):
    args = init_dist(args)
    fix_random_seeds(args.seed)

    with open(args.config_file, 'r') as f:
        cfg = json.loads(f.read())
    print_main("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    print_main("\n".join("%s: %s" % (k, str(v)) for k, v in cfg.items()))

    wandb.login(key="d4f10cd8f5ba2d005d10597e25acda8ca8eea9a5")
    logger = wandb.init(project=args.project_name,
                    group=args.experiment_name,
                    name=f'{args.experiment_name}_rank_{args.global_rank}',
                    job_type='train',
                    config=cfg,
                    notes='Experiment run.')

    # Instantiate the dataset
    dataset = DepthToSketchDataset(path_json_depth=cfg['data']['path_json_depth'],
                                         path_json_sketch=cfg['data']['path_json_sketch'], 
                                         path_meta=cfg['data']['path_meta'],
                                         resolution=cfg['processor']['resize'][0])
    dataset.transform = CondCvtrTransform(cfg=cfg['processor'])

    train_size = int(args.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f'Train dataset samples: {len(train_dataset)}')
    print(f'Train dataset samples: {len(val_dataset)}')

    val_dataloader = DataLoader(val_dataset,
                                batch_size=cfg['deepspeed']['train_micro_batch_size_per_gpu'],
                                shuffle=False,
                                sampler=DistributedSampler(val_dataset))

    model = CondCvtr(cfg).cuda()
    if has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    print_main(f'Model Device: {next(model.parameters()).device}')
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_params_count = sum(p.numel() for p in trainable_params)
    freezed_params_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print_main(f'Num trainable params: {trainable_params_count}')
    print_main(f'Num freezed params: {freezed_params_count}\n')

    # loss
    if cfg['losses']['type'] == 'CE':
        criterion = nn.CrossEntropyLoss()
    elif cfg['losses']['type'] == 'MSE':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f'Unsupported loss {cfg["losses"]["type"]}!')

    # optimizer
    if cfg['optimizer']['type'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(),
                                lr=cfg['optimizer']['lr'],
                                weight_decay=cfg['optimizer']['weight_decay'],
                                eps=cfg['optimizer']['eps'],
                                betas=cfg['optimizer']['betas'])
    else:
        raise ValueError(f'Unsupported optimizer {cfg["optimizer"]["type"]}!')

    # LR Scheduler
    steps_per_epoch = math.ceil(len(train_dataset) / (cfg['deepspeed']['train_micro_batch_size_per_gpu'] * args.world_size))
    print_main(f'steps_per_epoch: {steps_per_epoch}')
    if cfg['lr_scheduler']['type'] == 'WarmupCosineLR':
        lr_scheduler = WarmupCosineLR(
            optimizer=optimizer,
            base_lr=cfg['lr_scheduler']['base_lr'],
            total_steps=args.epochs * steps_per_epoch,
            start_lr=cfg['lr_scheduler']['start_lr'],
            end_lr=cfg['lr_scheduler']['end_lr'],
            warmup_steps=cfg['lr_scheduler']['warmup_epochs'] * steps_per_epoch,
        )
    else:
        ValueError(f'Unsupported LR scheduler {cfg["lr_scheduler"]["type"]}')

    # deepspeed
    model_engine, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        config=cfg['deepspeed'],
        optimizer=optimizer,
        training_data=train_dataset,
        lr_scheduler=lr_scheduler
    )
    # load checkpoint
    if args.load_ckpt_id >= 0:
        model_engine, start_epoch, best_val_loss, best_val_epoch = load_checkpoint_ds(model_engine, args)
        start_epoch += 1
    else:
        start_epoch = 0
        best_val_epoch = 0
        best_val_loss = np.float32('inf')

    # Start training
    model_engine.train()
    # logger.watch(model_engine, log="all", log_freq=100)
    print_main("Starting training!")
    for epoch in range(start_epoch, args.epochs):
        print_main(f'Start training epoch {epoch} ...')
        train_one_epoch(model_engine, criterion, train_dataloader, logger)
        print_main(f'Start validation epoch {epoch} ...')
        val_loss = validation_one_epoch(model_engine, criterion, val_dataloader, epoch, logger)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
            # save checkpoint, all processes need to call this function
            save_checkpoint_ds(model_engine, epoch, best_val_loss, best_val_epoch, args)

            # save model itself
            if args.global_rank == 0:
                # Check if using ZeRO-3 to decide how to get the state_dict
                if model_engine.zero_optimization_stage() == 3:
                    # For ZeRO-3, gather the full model state_dict
                    state_dict_last = get_fp32_state_dict_from_zero_checkpoint(args.checkpoint_dir)  # load latest checkpoint
                else:
                    # For other ZeRO stages or no ZeRO, directly access the model's state_dict
                    state_dict_last = copy.deepcopy(model_engine.module.state_dict())
                torch.save(state_dict_last, os.path.join(args.checkpoint_dir, f'model_{epoch}.pt'))
            print_main(f'Better validation performance obtained at epoch {epoch}, checkpoint saved.')
    print_main(f'Best validation performance obtained at epoch {best_val_epoch}, with loss {best_val_loss}.')

    # save final deepspeed checkpoint
    save_checkpoint_ds(model_engine, args.epochs - 1, best_val_loss, best_val_epoch, args)
    # save final model
    if args.global_rank == 0:
        # Check if using ZeRO-3 to decide how to get the state_dict
        if model_engine.zero_optimization_stage() == 3:
            # For ZeRO-3, gather the full model state_dict
            state_dict_last = get_fp32_state_dict_from_zero_checkpoint(args.checkpoint_dir)  # load latest checkpoint
        else:
            # For other ZeRO stages or no ZeRO, directly access the model's state_dict
            state_dict_last = copy.deepcopy(model_engine.module.state_dict())
        torch.save(state_dict_last, os.path.join(args.checkpoint_dir, 'model_last.pt'))


def train_one_epoch(model_engine, criterion, dataloader, logger):
    model_engine.train()  # Set the model in training mode
    for batch_index, batch in enumerate(dataloader):  # spectrum as input, ascan as label
        inputs = batch['depth_image'].to(model_engine.device)
        labels = batch['sketch_image'].to(model_engine.device)

        outputs = model_engine(inputs)
        loss = criterion(outputs, labels)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        model_engine.backward(loss)
        model_engine.step()

        logger.log({"step_loss": loss.item()})


def validation_one_epoch(model_engine, criterion, val_dataloader, epoch, logger):
    model_engine.eval()
    progress_bar = tqdm.tqdm(val_dataloader)
    eval_loss = 0
    nb_eval_steps = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(progress_bar):  # spectrum as input, ascan as label
            inputs = batch['depth_image'].to(model_engine.device)
            labels = batch['sketch_image'].to(model_engine.device)
            outputs = model_engine(inputs)
            tmp_eval_loss = criterion(outputs, labels)

            eval_loss += tmp_eval_loss.mean()
            nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    # Reduce to get the loss from all the GPU's
    dist.all_reduce(eval_loss)
    eval_loss = eval_loss.item() / dist.get_world_size()
    logger.log({"val_loss": eval_loss})
    print_main(f"Validation Loss for epoch {epoch} is: {eval_loss}")
    return eval_loss


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    train(args)
