"""
Paper: Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results.
Link: https://proceedings.neurips.cc/paper/2017/file/68053af2923e00204c3ca7c6a3150cf7-Paper.pdf
Code modified from: https://github.com/HiLab-git/SSL4MIS/blob/master/code/train_mean_teacher_2D.py
"""

import os
import torch

from tqdm import tqdm
from torchvision.utils import make_grid
from ._base import BaseTrainer
from ..builder import build_model


class MeanTeacherTrainer(BaseTrainer):

    def __init__(self, model=None,
                 ema_model=None,
                 optimizer=None,
                 scheduler=None,
                 criterions=None,
                 metrics=None,
                 logger=None,
                 device='cuda',
                 resume_from=None,
                 labeled_bs=12,
                 alpha=0.99,
                 consistency=1.0,
                 consistency_rampup=40.0,
                 data_parallel=False,
                 ckpt_save_path=None,
                 max_iter=60000,
                 eval_interval=1000,
                 save_image_interval=50,
                 save_ckpt_interval=2000) -> None:

        super().__init__(model, optimizer, scheduler, criterions, metrics, logger, device, resume_from, labeled_bs,
                         consistency, consistency_rampup, data_parallel, ckpt_save_path, max_iter, eval_interval,
                         save_image_interval, save_ckpt_interval)

        self.ema_model = build_model(ema_model).to(device)
        for param in self.ema_model.parameters():
            param.detach_()

        self.alpha = alpha

    def train_step(self, batch_data, step, save_image):
        log_infos, scalars, images = {}, {}, {}
        data, label = batch_data['image'].to(self.device), batch_data['label'].to(self.device)

        unlabeled_data = data[self.labeled_bs:]

        noise = torch.clamp(torch.randn_like(unlabeled_data) * 0.1, -0.2, 0.2)
        ema_inputs = unlabeled_data + noise

        outputs = self.model(data)
        outputs_soft = torch.softmax(outputs['seg'], dim=1)
        with torch.no_grad():
            ema_output = self.ema_model(ema_inputs)
            ema_output_soft = torch.softmax(ema_output['seg'], dim=1)

        supervised_loss = 0.
        for criterion in self.criterions:
            loss_ = criterion(outputs['seg'][:self.labeled_bs], label[:self.labeled_bs])
            supervised_loss += loss_
            log_infos[criterion.name] = float(format(loss_, '.5f'))
            scalars[f'loss/{criterion.name}'] = loss_

        consistency_weight = self.get_current_consistency_weight(step // 150)
        if step < 1000:
            consistency_loss = 0.0
        else:
            consistency_loss = torch.mean((outputs_soft[self.labeled_bs:] - ema_output_soft) ** 2)

        loss = supervised_loss + consistency_weight * consistency_loss

        log_infos['con_weight'] = float(format(consistency_weight, '.5f'))
        log_infos['loss_con'] = float(format(consistency_loss, '.5f'))
        log_infos['loss'] = float(format(loss, '.5f'))
        scalars['consistency_weight'] = consistency_weight
        scalars['loss/loss_consistency'] = consistency_loss
        scalars['loss/total'] = loss

        preds = torch.argmax(outputs['seg'], dim=1, keepdim=True).to(torch.float)

        metric_res = self.metrics[0](preds, label)
        for key in metric_res.keys():
            log_infos[f'{self.metrics[0].name}.{key}'] = float(format(metric_res[key], '.5f'))
            scalars[f'train/{self.metrics[0].name}.{key}'] = metric_res[key]

        if save_image:
            grid_image = make_grid(data, 4, normalize=True)
            images['train/images'] = grid_image
            grid_image = make_grid(preds * 50., 4, normalize=False)
            images['train/preds'] = grid_image
            grid_image = make_grid(label * 50., 4, normalize=False)
            images['train/ground_truth'] = grid_image
        return loss, log_infos, scalars, images

    def train(self, train_loader, val_loader):
        max_epoch = self.max_iter // len(train_loader) + 1
        step = self.start_step
        with tqdm(total=self.max_iter - self.start_step, bar_format='[{elapsed}<{remaining}] ') as pbar:
            for _ in range(max_epoch):
                for batch_data in train_loader:
                    save_image = True if (step + 1) % self.save_image_interval == 0 else False

                    loss, log_infos, scalars, images = self.train_step(batch_data, step, save_image)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    self.update_ema_variables(step)

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

    def update_ema_variables(self, global_step):
        alpha = min(1 - 1 / (global_step + 1), self.alpha)
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
