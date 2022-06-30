import os
import yaml
import argparse
import warnings
import random

import torch
import numpy as np

from datetime import datetime
from codes.builder import build_dataloader, build_logger
from codes.utils.utils import Namespace, parse_yaml
from codes.trainers import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/comparison_acdc_224_136/ugpcl_unet_r50.yaml',
                        help='train config file path: xxx.yaml')
    parser.add_argument('--work_dir', type=str,
                        default=f'results/comparison_acdc_224_136',
                        help='the dir to save logs and models')
    parser.add_argument('--resume_from', type=str,
                        # default='results/comparison_acdc_224_136/ugcl_mem_unet_r50_0430155558/iter_1000.pth',
                        default=None,
                        help='the checkpoint file to resume from')
    parser.add_argument('--start_step', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data_parallel', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--deterministic', type=bool, default=True,
                        help='whether to set deterministic options for CUDNN backend.')
    args = parser.parse_args()

    args_dict = parse_yaml(args.config)

    for key, value in Namespace(args_dict).__dict__.items():
        if key in ['name', 'dataset', 'train', 'logger']:
            vars(args)[key] = value

    for key, value in Namespace(args_dict).__dict__.items():
        if key not in ['name', 'dataset', 'train', 'logger']:
            vars(args.train.kwargs)[key] = value

    if args.work_dir is None:
        args.work_dir = f'results/{args.dataset.name}'
    if args.resume_from is not None:
        args.logger.log_dir = os.path.split(os.path.abspath(args.resume_from))[0]
        args.logger.file_mode = 'a'
    else:
        args.logger.log_dir = f'{args.work_dir}/{args.name}_{datetime.now().strftime("%m%d%H%M%S")}'
    args.ckpt_save_path = args.logger.log_dir

    for key in args.__dict__.keys():
        if key not in args_dict.keys():
            args_dict[key] = args.__dict__[key]

    return args, args_dict


def set_deterministic(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_trainer(name,
                  logger=None,
                  device='cuda',
                  data_parallel=False,
                  ckpt_save_path=None,
                  resume_from=None,
                  **kwargs):
    return eval(f'{name}')(logger=logger, device=device, data_parallel=data_parallel, ckpt_save_path=ckpt_save_path,
                           resume_from=resume_from, **kwargs)


def train():
    args, args_dict = get_args()
    set_deterministic(args.seed)

    def worker_init_fn(worker_id):
        random.seed(worker_id + args.seed)

    train_loader, val_loader = build_dataloader(args.dataset, worker_init_fn)
    logger = build_logger(args.logger)

    args_yaml_info = yaml.dump(args_dict, sort_keys=False, default_flow_style=None)
    yaml_file_name = os.path.split(args.config)[-1]
    with open(os.path.join(args.ckpt_save_path, yaml_file_name), 'w') as f:
        f.write(args_yaml_info)
        f.close()

    logger.info(f'\n{args_yaml_info}\n')

    trainer = build_trainer(name=args.train.name,
                            logger=logger,
                            device=args.device,
                            data_parallel=args.data_parallel,
                            ckpt_save_path=args.ckpt_save_path,
                            resume_from=args.resume_from,
                            **args.train.kwargs.__dict__)
    trainer.train(train_loader, val_loader)
    logger.close()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    train()
