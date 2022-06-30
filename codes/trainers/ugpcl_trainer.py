import os
import random
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from torch import nn
from tqdm import tqdm
from prettytable import PrettyTable
from colorama import Fore
from torchvision.utils import make_grid
from ._base import BaseTrainer
from ..utils import ramps
from ..losses import PixelContrastLoss


class UGPCLTrainer(BaseTrainer):

    def __init__(self,
                 model=None,
                 optimizer=None,
                 scheduler=None,
                 criterions=None,
                 metrics=None,
                 logger=None,
                 device='cuda',
                 resume_from=None,
                 labeled_bs=8,
                 data_parallel=False,
                 ckpt_save_path=None,
                 max_iter=6000,
                 eval_interval=1000,
                 save_image_interval=50,
                 save_ckpt_interval=2000,
                 consistency=0.1,
                 consistency_rampup=40.0,
                 tf_decoder_weight=0.4,
                 cls_weight=0.1,
                 contrast_type='ugpcl',  # ugpcl, pseudo, sup, none
                 contrast_weight=0.1,
                 temperature=0.1,
                 base_temperature=0.07,
                 max_samples=1024,
                 max_views=1,
                 memory=True,
                 memory_size=100,
                 pixel_update_freq=10,
                 pixel_classes=4,
                 dim=256) -> None:

        super(UGPCLTrainer, self).__init__(model, optimizer, scheduler, criterions, metrics, logger, device,
                                           resume_from, labeled_bs, consistency, consistency_rampup, data_parallel,
                                           ckpt_save_path, max_iter, eval_interval, save_image_interval,
                                           save_ckpt_interval)

        self.tf_decoder_weight = tf_decoder_weight
        self.cls_weight = cls_weight
        self.cls_criterion = torch.nn.CrossEntropyLoss()

        self.contrast_type = contrast_type
        self.contrast_weight = contrast_weight
        self.contrast_criterion = PixelContrastLoss(temperature=temperature,
                                                    base_temperature=base_temperature,
                                                    max_samples=max_samples,
                                                    max_views=max_views,
                                                    device=device)
        # memory param
        self.memory = memory
        self.memory_size = memory_size
        self.pixel_update_freq = pixel_update_freq

        if self.memory:
            self.segment_queue = torch.randn(pixel_classes, self.memory_size, dim)
            self.segment_queue = nn.functional.normalize(self.segment_queue, p=2, dim=2)
            self.segment_queue_ptr = torch.zeros(pixel_classes, dtype=torch.long)
            self.pixel_queue = torch.zeros(pixel_classes, self.memory_size, dim)
            self.pixel_queue = nn.functional.normalize(self.pixel_queue, p=2, dim=2)
            self.pixel_queue_ptr = torch.zeros(pixel_classes, dtype=torch.long)

    def _dequeue_and_enqueue(self, keys, labels):
        batch_size = keys.shape[0]
        feat_dim = keys.shape[1]

        labels = torch.nn.functional.interpolate(labels, (keys.shape[2], keys.shape[3]), mode='nearest')

        for bs in range(batch_size):
            this_feat = keys[bs].contiguous().view(feat_dim, -1)
            this_label = labels[bs].contiguous().view(-1)
            this_label_ids = torch.unique(this_label)
            this_label_ids = [x for x in this_label_ids if x > 0]
            for lb in this_label_ids:
                idxs = (this_label == lb).nonzero()
                lb = int(lb.item())
                # segment enqueue and dequeue
                feat = torch.mean(this_feat[:, idxs], dim=1).squeeze(1)
                ptr = int(self.segment_queue_ptr[lb])
                self.segment_queue[lb, ptr, :] = nn.functional.normalize(feat.view(-1), p=2, dim=0)
                self.segment_queue_ptr[lb] = (self.segment_queue_ptr[lb] + 1) % self.memory_size

                # pixel enqueue and dequeue
                num_pixel = idxs.shape[0]
                perm = torch.randperm(num_pixel)
                K = min(num_pixel, self.pixel_update_freq)
                feat = this_feat[:, perm[:K]]
                feat = torch.transpose(feat, 0, 1)
                ptr = int(self.pixel_queue_ptr[lb])

                if ptr + K >= self.memory_size:
                    self.pixel_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                    self.pixel_queue_ptr[lb] = 0
                else:
                    self.pixel_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                    self.pixel_queue_ptr[lb] = (self.pixel_queue_ptr[lb] + 1) % self.memory_size

    @staticmethod
    def _random_rotate(image, label):
        angle = float(torch.empty(1).uniform_(-20., 20.).item())
        image = TF.rotate(image, angle)
        label = TF.rotate(label, angle)
        return image, label

    def train_step(self, batch_data, step, save_image):
        log_infos, scalars = {}, {}
        images = {}
        data_, label_ = batch_data['image'].to(self.device), batch_data['label'].to(self.device)
        # data, label = self._random_aug(data_, label_)
        if self.cls_weight >= 0.:
            images_, labels_ = [], []
            cls_label = []
            for image, label in zip(data_, label_):
                rot_times = random.randrange(0, 4)
                cls_label.append(rot_times)
                image = torch.rot90(image, rot_times, [1, 2])
                label = torch.rot90(label, rot_times, [1, 2])
                image, label = self._random_rotate(image, label)
                images_.append(image)
                labels_.append(label)
            cls_label = torch.tensor(cls_label).to(self.device)
            data = torch.stack(images_, dim=0).to(self.device)
            label = torch.stack(labels_, dim=0).to(self.device)
        else:
            data = data_
            label = label_
            cls_label = None

        outputs = self.model(data, self.device)
        seg = outputs['seg']
        seg_tf = outputs['seg_tf']

        supervised_loss = 0.
        for criterion in self.criterions:
            loss_ = criterion(seg[:self.labeled_bs], label[:self.labeled_bs]) + \
                    self.tf_decoder_weight * criterion(seg_tf[:self.labeled_bs], label[:self.labeled_bs])
            supervised_loss += loss_
            log_infos[criterion.name] = float(format(loss_, '.5f'))
            scalars[f'loss/{criterion.name}'] = loss_

        loss_cls = self.cls_criterion(outputs['cls'], cls_label) if self.cls_weight > 0. else 0.

        seg_soft = torch.softmax(seg, dim=1)
        seg_tf_soft = torch.softmax(seg_tf, dim=1)
        consistency_weight = self.get_current_consistency_weight(step // 100)
        consistency_loss = torch.mean((seg_soft[self.labeled_bs:] - seg_tf_soft[self.labeled_bs:]) ** 2)

        loss = supervised_loss + consistency_weight * consistency_loss + self.cls_weight * loss_cls

        log_infos['loss_cls'] = float(format(loss_cls, '.5f'))
        log_infos['con_weight'] = float(format(consistency_weight, '.5f'))
        log_infos['loss_con'] = float(format(consistency_loss, '.5f'))
        log_infos['loss'] = float(format(loss, '.5f'))
        scalars['loss/loss_cls'] = loss_cls
        scalars['consistency_weight'] = consistency_weight
        scalars['loss/loss_consistency'] = consistency_loss
        scalars['loss/total'] = loss

        preds = torch.argmax(seg_soft, dim=1, keepdim=True).to(torch.float)

        log_infos['loss_contrast'] = 0.
        scalars['loss/contrast'] = 0.
        if step > 1000 and self.contrast_weight > 0.:
            # queue = torch.cat((self.segment_queue, self.pixel_queue), dim=1) if self.memory else None
            queue = self.segment_queue if self.memory else None
            if self.contrast_type == 'ugpcl':
                seg_mean = torch.mean(torch.stack([F.softmax(seg, dim=1), F.softmax(seg_tf, dim=1)]), dim=0)
                uncertainty = -1.0 * torch.sum(seg_mean * torch.log(seg_mean + 1e-6), dim=1, keepdim=True)
                threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(step, self.max_iter)) * np.log(2)
                uncertainty_mask = (uncertainty > threshold)
                mean_preds = torch.argmax(F.softmax(seg_mean, dim=1).detach(), dim=1, keepdim=True).float()
                certainty_pseudo = mean_preds.clone()
                certainty_pseudo[uncertainty_mask] = -1
                certainty_pseudo[:self.labeled_bs] = label[:self.labeled_bs]
                contrast_loss = self.contrast_criterion(outputs['embed'], certainty_pseudo, preds, queue=queue)
                scalars['uncertainty_rate'] = torch.sum(uncertainty_mask == True) / \
                                              (torch.sum(uncertainty_mask == True) + torch.sum(
                                                  uncertainty_mask == False))
                if self.memory:
                    self._dequeue_and_enqueue(outputs['embed'].detach(), certainty_pseudo.detach())
                if save_image:
                    grid_image = make_grid(mean_preds * 50., 4, normalize=False)
                    images['train/mean_preds'] = grid_image
                    grid_image = make_grid(certainty_pseudo * 50., 4, normalize=False)
                    images['train/certainty_pseudo'] = grid_image
                    grid_image = make_grid(uncertainty, 4, normalize=False)
                    images['train/uncertainty'] = grid_image
                    grid_image = make_grid(uncertainty_mask.float(), 4, normalize=False)
                    images['train/uncertainty_mask'] = grid_image
            elif self.contrast_type == 'pseudo':
                contrast_loss = self.contrast_criterion(outputs['embed'], preds.detach(), preds, queue=queue)
                if self.memory:
                    self._dequeue_and_enqueue(outputs['embed'].detach(), preds.detach())
            elif self.contrast_type == 'sup':
                contrast_loss = self.contrast_criterion(outputs['embed'][:self.labeled_bs], label[:self.labeled_bs],
                                                        preds[:self.labeled_bs], queue=queue)
                if self.memory:
                    self._dequeue_and_enqueue(outputs['embed'].detach()[:self.labeled_bs],
                                              label.detach()[:self.labeled_bs])
            else:
                contrast_loss = 0.
            loss += self.contrast_weight * contrast_loss
            log_infos['loss_contrast'] = float(format(contrast_loss, '.5f'))
            scalars['loss/contrast'] = contrast_loss

        tf_preds = torch.argmax(seg_tf_soft, dim=1, keepdim=True).to(torch.float)
        metric_res = self.metrics[0](preds, label)
        for key in metric_res.keys():
            log_infos[f'{self.metrics[0].name}.{key}'] = float(format(metric_res[key], '.5f'))
            scalars[f'train/{self.metrics[0].name}.{key}'] = metric_res[key]

        if save_image:
            grid_image = make_grid(data, 4, normalize=True)
            images['train/images'] = grid_image
            grid_image = make_grid(preds * 50., 4, normalize=False)
            images['train/preds'] = grid_image
            grid_image = make_grid(tf_preds * 50., 4, normalize=False)
            images['train/tf_preds'] = grid_image
            grid_image = make_grid(label * 50., 4, normalize=False)
            images['train/labels'] = grid_image

        return loss, log_infos, scalars, images

    def val_step(self, batch_data):
        data, labels = batch_data['image'].to(self.device), batch_data['label'].to(self.device)
        preds = self.model.inference(data)
        metric_total_res = {}
        for metric in self.metrics:
            metric_total_res[metric.name] = metric(preds, labels)
        return metric_total_res

    def val_step_tf(self, batch_data):
        data, labels = batch_data['image'].to(self.device), batch_data['label'].to(self.device)
        preds = self.model.inference_tf(data, self.device)
        metric_total_res = {}
        for metric in self.metrics:
            metric_total_res[metric.name] = metric(preds, labels)
        return metric_total_res

    @torch.no_grad()
    def val_tf(self, val_loader, test=False):
        self.model.eval()
        val_res = None
        val_scalars = {}
        if self.logger is not None:
            self.logger.info('Evaluating...')
        if test:
            val_loader = tqdm(val_loader, desc='Testing', unit='batch',
                              bar_format='%s{l_bar}{bar}{r_bar}%s' % (Fore.LIGHTCYAN_EX, Fore.RESET))
        for batch_data in val_loader:
            batch_res = self.val_step_tf(batch_data)  # {'Dice':{'c1':0.1, 'c2':0.1, ...}, ...}
            if val_res is None:
                val_res = batch_res
            else:
                for metric_name in val_res.keys():
                    for key in val_res[metric_name].keys():
                        val_res[metric_name][key] += batch_res[metric_name][key]
        for metric_name in val_res.keys():
            for key in val_res[metric_name].keys():
                val_res[metric_name][key] = val_res[metric_name][key] / len(val_loader)
                val_scalars[f'val_tf/{metric_name}.{key}'] = val_res[metric_name][key]

            val_res_list = [_.cpu() for _ in val_res[metric_name].values()]
            val_res[metric_name]['Mean'] = np.mean(val_res_list[1:])
            val_scalars[f'val_tf/{metric_name}.Mean'] = val_res[metric_name]['Mean']

        val_table = PrettyTable()
        val_table.field_names = ['Metirc'] + list(list(val_res.values())[0].keys())
        for metric_name in val_res.keys():
            if metric_name in ['Dice', 'Jaccard', 'Acc', 'IoU', 'Recall', 'Precision']:
                temp = [float(format(_ * 100, '.2f')) for _ in val_res[metric_name].values()]
            else:
                temp = [float(format(_, '.2f')) for _ in val_res[metric_name].values()]
            val_table.add_row([metric_name] + temp)
        return val_res, val_scalars, val_table

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

                            val_res, val_scalars, val_table = self.val_tf(val_loader)
                            self.logger.info(f'val_tf result:\n{val_table.get_string()}')
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
