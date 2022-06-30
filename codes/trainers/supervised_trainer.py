import torch
from ._base import BaseTrainer
from torchvision.utils import make_grid


class SupervisedTrainer(BaseTrainer):

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
                 data_parallel=False,
                 ckpt_save_path=None,
                 max_iter=10000,
                 eval_interval=1000,
                 save_image_interval=50,
                 save_ckpt_interval=2000) -> None:

        super().__init__(model, optimizer, scheduler, criterions, metrics, logger, device, resume_from, labeled_bs,
                         1.0, 40.0, data_parallel, ckpt_save_path, max_iter, eval_interval,
                         save_image_interval, save_ckpt_interval)

    def train_step(self, batch_data, step, save_image):
        log_infos, scalars, images = {}, {}, {}
        data, label = batch_data['image'].to(self.device), batch_data['label'].to(self.device)
        logits = self.model(data)
        loss = 0.
        for criterion in self.criterions:
            loss_ = criterion(logits['seg'][:self.labeled_bs], label[:self.labeled_bs])
            loss += loss_
            log_infos[criterion.name] = float(format(loss_, '.5f'))
            scalars[f'loss/{criterion.name}'] = loss_

        log_infos['loss'] = float(format(loss, '.5f'))
        scalars['loss/total'] = loss
        preds = torch.argmax(torch.softmax(logits['seg'], dim=1), dim=1, keepdim=True).to(torch.float)

        metric_res = self.metrics[0](preds, label)
        for key in metric_res.keys():
            log_infos[f'{self.metrics[0].name}.{key}'] = float(format(metric_res[key], '.5f'))
            scalars[f'train/{self.metrics[0].name}.{key}'] = metric_res[key]

        images = {}
        if save_image:
            grid_image = make_grid(data[:self.labeled_bs], 4, normalize=True)
            images['train/image'] = grid_image
            grid_image = make_grid(preds[:self.labeled_bs] * 50., 4, normalize=False)
            images['train/pred'] = grid_image
            grid_image = make_grid(label[:self.labeled_bs] * 50., 4, normalize=False)
            images['train/label'] = grid_image
        return loss, log_infos, scalars, images