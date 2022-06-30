import os
from abc import abstractmethod

import numpy as np
import torch

from tqdm import tqdm
from prettytable import PrettyTable
from colorama import Fore
from ..utils import ramps
from ..builder import _build_from_cfg, build_model, build_optimizer, build_scheduler


class BaseTrainer:

    def __init__(self,
                 model=None,
                 optimizer=None,
                 scheduler=None,
                 criterions=None,
                 metrics=None,
                 logger=None,
                 device='cuda',
                 resume_from=None,
                 labeled_bs=12,
                 consistency=1.0,
                 consistency_rampup=40.0,
                 data_parallel=False,
                 ckpt_save_path=None,
                 max_iter=10000,
                 eval_interval=1000,
                 save_image_interval=50,
                 save_ckpt_interval=2000) -> None:
        super(BaseTrainer, self).__init__()
        self.model = None
        # build cfg
        if model is not None:
            self.model = build_model(model).to(device)
        if optimizer is not None:
            self.optimizer = build_optimizer(self.model.parameters(), optimizer)
        if scheduler is not None:
            self.scheduler = build_scheduler(self.optimizer, scheduler)
        self.criterions = []
        if criterions is not None:
            for criterion_cfg in criterions:
                self.criterions.append(_build_from_cfg(criterion_cfg))
        self.metrics = []
        if metrics is not None:
            for metric_cfg in metrics:
                self.metrics.append(_build_from_cfg(metric_cfg))

        # semi-supervised params
        self.labeled_bs = labeled_bs
        self.consistency = consistency
        self.consistency_rampup = consistency_rampup

        # train params
        self.logger = logger
        self.device = device
        self.data_parallel = data_parallel
        self.ckpt_save_path = ckpt_save_path

        self.max_iter = max_iter
        self.eval_interval = eval_interval
        self.save_image_interval = save_image_interval
        self.save_ckpt_interval = save_ckpt_interval

        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1])

        if resume_from is not None:
            ckpt = torch.load(resume_from)
            model.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
            scheduler.load_state_dict(ckpt['scheduler'])
            self.start_step = ckpt['step']

            logger.info(f'Resume from {resume_from}.')
            logger.info(f'Train from step {self.start_step}.')
        else:
            self.start_step = 0
            if self.model is not None:
                logger.info(f'\n{self.model}\n')

        logger.info(f'start training...')

    @abstractmethod
    def train_step(self, batch_data, step, save_image):
        loss = 0.
        log_infos, scalars, images = {}, {}, {}
        return loss, log_infos, scalars, images

    def val_step(self, batch_data):
        data, labels = batch_data['image'].to(self.device), batch_data['label'].to(self.device)
        preds = self.model.inference(data)
        metric_total_res = {}
        for metric in self.metrics:
            metric_total_res[metric.name] = metric(preds, labels)
        return metric_total_res

    def train(self, train_loader, val_loader):
        # iter_train_loader = iter(train_loader)
        max_epoch = self.max_iter // len(train_loader) + 1
        step = self.start_step
        self.model.train()
        with tqdm(total=self.max_iter - self.start_step, bar_format='[{elapsed}<{remaining}] ') as pbar:
            for _ in range(max_epoch):
                for batch_data in train_loader:
                    save_image = True if (step + 1) % self.save_image_interval == 0 else False

                    loss, log_infos, scalars, images = self.train_step(batch_data, step, save_image)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    if (step + 1) % 10 == 0:
                        scalars.update({'lr': self.scheduler.get_lr()[0]})
                        log_infos.update({'lr': self.scheduler.get_lr()[0]})
                        self.logger.update_scalars(scalars, step + 1)
                        self.logger.info(f'[{step + 1}/{self.max_iter}] {log_infos}')

                    if save_image:
                        self.logger.update_images(images, step + 1)

                    if (step + 1) % self.eval_interval == 0:
                        if val_loader is not None:
                            val_res, val_scalars, val_table = self.val(val_loader)
                            self.logger.info(f'val result:\n{val_table.get_string()}')
                            self.logger.update_scalars(val_scalars, step + 1)
                            self.model.train()

                    if (step + 1) % self.save_ckpt_interval == 0:
                        if not os.path.exists(self.ckpt_save_path):
                            os.makedirs(self.ckpt_save_path)
                        self.save_ckpt(step + 1, f'{self.ckpt_save_path}/iter_{step + 1}.pth')
                    step += 1
                    pbar.update(1)
                    if step >= self.max_iter:
                        break
                if step >= self.max_iter:
                    break

        if not os.path.exists(self.ckpt_save_path):
            os.makedirs(self.ckpt_save_path)
            torch.save(self.model.state_dict(), f'{self.ckpt_save_path}/ckpt_final.pth')

    @torch.no_grad()
    def val(self, val_loader, test=False):
        self.model.eval()
        val_res = None
        val_scalars = {}
        if self.logger is not None:
            self.logger.info('Evaluating...')
        if test:
            val_loader = tqdm(val_loader, desc='Testing', unit='batch',
                              bar_format='%s{l_bar}{bar}{r_bar}%s' % (Fore.LIGHTCYAN_EX, Fore.RESET))
        for batch_data in val_loader:
            batch_res = self.val_step(batch_data)  # {'Dice':{'c1':0.1, 'c2':0.1, ...}, ...}
            if val_res is None:
                val_res = batch_res
            else:
                for metric_name in val_res.keys():
                    for key in val_res[metric_name].keys():
                        val_res[metric_name][key] += batch_res[metric_name][key]
        for metric_name in val_res.keys():
            for key in val_res[metric_name].keys():
                val_res[metric_name][key] = val_res[metric_name][key] / len(val_loader)
                val_scalars[f'val/{metric_name}.{key}'] = val_res[metric_name][key]

            val_res_list = [_.cpu() for _ in val_res[metric_name].values()]
            val_res[metric_name]['Mean'] = np.mean(val_res_list[1:])
            val_scalars[f'val/{metric_name}.Mean'] = val_res[metric_name]['Mean']

        val_table = PrettyTable()
        val_table.field_names = ['Metirc'] + list(list(val_res.values())[0].keys())
        for metric_name in val_res.keys():
            if metric_name in ['Dice', 'Jaccard', 'Acc', 'IoU', 'Recall', 'Precision']:
                temp = [float(format(_ * 100, '.2f')) for _ in val_res[metric_name].values()]
            else:
                temp = [float(format(_, '.2f')) for _ in val_res[metric_name].values()]
            val_table.add_row([metric_name] + temp)
        return val_res, val_scalars, val_table

    def get_current_consistency_weight(self, epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return self.consistency * ramps.sigmoid_rampup(epoch, self.consistency_rampup)

    def save_ckpt(self, step, save_path):
        ckpt = {'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'step': step}
        torch.save(ckpt, save_path)
        self.logger.info('Checkpoint saved!')
