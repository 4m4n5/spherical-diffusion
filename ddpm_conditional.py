"""
This code train a conditional diffusion model on CIFAR.
It is based on @dome272.
@wandbcode{condition_diffusion}
"""

import argparse
from contextlib import nullcontext
import os
import copy
import numpy as np
from utils import set_seed
import torch
from torch import optim
import torch.nn as nn
from types import SimpleNamespace
from fastprogress import progress_bar, master_bar
import utils
from utils import *
from data import get_circle_dot_data, create_loader, create_sampler, get_nuclei_mask_data
from modules import UNet_conditional, EMA
import logging
import wandb
from diffusion import Diffusion
import random
import torch.backends.cudnn as cudnn

def process_input_mask(labels, device, num_classes=7):
    labels = labels.long()
    bs, _, h, w = labels.size()
    nc = num_classes
    input_label = torch.FloatTensor(bs, nc, h, w).zero_().to(device)
    input_semantics = input_label.scatter_(1, labels, 1.0)

    # One-hot 
    cond = torch.sum(torch.sum(input_semantics, dim=2), dim=2).bool().int().float()

    return input_semantics, cond


def cycle(dataloader, start_iteration: int = 0):
    r"""
    A generator to yield batches of data from dataloader infinitely.
    Internally, it sets the ``epoch`` for dataloader sampler to shuffle the
    examples. One may optionally provide the starting iteration to make sure
    the shuffling seed is different and continues naturally.
    """
    iteration = start_iteration

    while True:
        if isinstance(dataloader.sampler, torch.utils.data.DistributedSampler):
            # Set the `epoch` of DistributedSampler as current iteration. This
            # is a way of determinisitic shuffling after every epoch, so it is
            # just a seed and need not necessarily be the "epoch".
            dataloader.sampler.set_epoch(iteration)

        for batch in dataloader:
            yield batch
            iteration += 1


config = SimpleNamespace(    
    run_name = "mask_gen",
    noise_steps=1000,
    seed = 42,
    batch_size = 2,
    img_size = 128,
    num_classes = 7,
    c_in = 7,
    c_out = 7,
    dataset_path = r"/home/as3ek/data/lizard_split_norm_bright3/",
    device = "cuda",
    slice_size = 1,
    use_wandb = True,
    do_validation = False,
    fp16 = True,
    num_workers=4,
    lr = 2e-4,
    log_interval = 200,
    save_interval = 50,
    num_steps = 100000,
    warmup_steps = 500,
    is_train = True,
)


logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


def parse_args(config):
    parser = argparse.ArgumentParser(description='Process hyper-parameters')
    parser.add_argument('--run_name', type=str, default=config.run_name, help='name of the run')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='batch size')
    parser.add_argument('--img_size', type=int, default=config.img_size, help='image size')
    parser.add_argument('--num_classes', type=int, default=config.num_classes, help='number of classes')
    parser.add_argument('--dataset_path', type=str, default=config.dataset_path, help='path to dataset')
    parser.add_argument('--use_wandb', type=bool, default=config.use_wandb, help='use wandb')
    parser.add_argument('--is_train', type=bool, default=config.is_train, help='train mode')
    parser.add_argument('--lr', type=float, default=config.lr, help='learning rate')
    parser.add_argument('--slice_size', type=int, default=config.slice_size, help='slice size')
    parser.add_argument('--noise_steps', type=int, default=config.noise_steps, help='noise steps')
    parser.add_argument('--num_steps', type=int, default=config.num_steps, help='num training steps')
    parser.add_argument('--warmup_steps', type=int, default=config.warmup_steps, help='num warmup steps')
    parser.add_argument('--log_interval', type=int, default=config.log_interval, help='num training steps')
    parser.add_argument('--save_interval', type=int, default=config.save_interval, help='num training steps')
    parser.add_argument('--c_in', type=int, default=config.c_in, help='c_in')
    parser.add_argument('--c_out', type=int, default=config.c_out, help='c_out')
    # parser.add_argument('--config', default='./configs/pretrain.yaml')
    # parser.add_argument('--output_dir', default='output/Pretrain')  
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--ema_checkpoint', default='')

    # Distributed training parameters    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = vars(parser.parse_args())
    
    # update config with parsed args
    for k, v in args.items():
        setattr(config, k, v)

def main(config):
    utils.init_distributed_mode(config)
    device = torch.device(config.device)

    # fix the seed for reproducibility
    seed = config.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False

    if utils.is_main_process():
        wandb.init(project="spherical_diffusion", config=vars(config))

    #### Dataset #### 
    print("Creating dataset")
    datasets = [get_nuclei_mask_data(config)]
    print('number of training samples: %d'%len(datasets[0]))

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    samplers = create_sampler(datasets, [True], num_tasks, global_rank)         
    data_loader = create_loader(
        datasets, 
        samplers, 
        batch_size=[config.batch_size], num_workers=[config.num_workers], is_trains=[config.is_train], collate_fns=[None]
    )[0]

    start_iteration = 0
    train_dataloader_iter = cycle(data_loader, start_iteration)

    #### Model ####
    print("Creating model") 
    diffuser = Diffusion(config.noise_steps, img_size=config.img_size, num_classes=config.num_classes, c_in=config.c_in, c_out=config.c_out)
    # ema = EMA(0.995)
    
    mk_folders(config.run_name)
    
    optimizer = optim.AdamW(diffuser.model.parameters(), lr=config.lr, weight_decay=0.001)
    mse = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()

    # Load checkpoint
    if config.checkpoint != '':
        diffuser.load(config.checkpoint)

    # model_without_ddp = diffuser.model
    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(diffuser.model, device_ids=[config.gpu])
        # model_without_ddp = model.module

    # Start training
    diffuser.model.train()
    avg_loss = 0
    for iteration in range(start_iteration + 1, config.num_steps + 1):
        optimizer.zero_grad()
        batch = next(train_dataloader_iter)

        if iteration <= config.warmup_steps:
            warmup_lr_schedule(optimizer, iteration, config.warmup_steps, 0, config.lr)
        
        # if iteration > config.warmup_steps:
        #     cosine_lr_schedule(optimizer, iteration - config.warmup_steps, config.num_steps - config.warmup_steps, config.lr, config.lr/100.0)

        with torch.autocast("cuda") and torch.enable_grad():
            masks = batch.to(device)

            pmask, cond = process_input_mask(masks, device)
            pmask = 2.0 * pmask - 1.0

            t = diffuser.sample_timesteps(pmask.shape[0]).to(device)
            x_t, noise = diffuser.noise_images(pmask, t)

            if config.checkpoint != '':
                if utils.is_main_process():
                    diffuser.log_images(cond=cond, iteration=iteration)
            else:
                if np.random.random() < 0.1:
                    cond = None
                
                predicted_noise = diffuser.model(x_t, t, cond)
                loss = mse(noise, predicted_noise)
                avg_loss += loss
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # ema.step_ema(diffuser.ema_model, diffuser.model)

                if iteration % config.log_interval == 0:
                    if config.use_wandb:
                        if utils.is_main_process():
                            wandb.log({
                                "loss": loss.item(),
                                "learning_rate": optimizer.param_groups[0]['lr'], 
                            })
                if iteration % config.save_interval == 0:
                    if utils.is_main_process():
                        diffuser.save_model(config.run_name, optimizer=optimizer, iteration=iteration, use_wandb=config.use_wandb)
                        # diffuser.log_images(use_wandb=config.use_wandb)

                    torch.distributed.barrier()


if __name__ == '__main__':
    parse_args(config)

    ## seed everything
    set_seed(config.seed)
    main(config)
