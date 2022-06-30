import torch
import torch.nn.functional as F

from abc import ABC
from torch import nn


class PixelContrastLoss(nn.Module, ABC):
    def __init__(self,
                 temperature=0.07,
                 base_temperature=0.07,
                 max_samples=1024,
                 max_views=100,
                 ignore_index=-1,
                 device='cuda:0'):
        super(PixelContrastLoss, self).__init__()

        self.temperature = temperature
        self.base_temperature = base_temperature

        self.ignore_label = ignore_index

        self.max_samples = max_samples
        self.max_views = max_views

        self.device = device

    def _anchor_sampling(self, X, y_hat, y):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]

            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)
        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).to(self.device)
        y_ = torch.zeros(total_classes, dtype=torch.float).to(self.device)

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    print('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1
        return X_, y_

    def _sample_negative(self, Q):
        class_num, memory_size, feat_size = Q.shape

        x_ = torch.zeros((class_num * memory_size, feat_size)).float().to(self.device)
        y_ = torch.zeros((class_num * memory_size, 1)).float().to(self.device)

        sample_ptr = 0
        for c in range(class_num):
            if c == 0:
                continue
            this_q = Q[c, :memory_size, :]
            x_[sample_ptr:sample_ptr + memory_size, ...] = this_q
            y_[sample_ptr:sample_ptr + memory_size, ...] = c
            sample_ptr += memory_size
        return x_, y_

    def _contrastive(self, X_anchor, y_anchor, queue=None):
        anchor_num, n_view = X_anchor.shape[0], X_anchor.shape[1]

        y_anchor = y_anchor.contiguous().view(-1, 1)  # (anchor_num × n_view) × 1
        anchor_count = n_view
        anchor_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)  # (anchor_num × n_view) × feat_dim

        if queue is not None:
            X_contrast, y_contrast = self._sample_negative(queue)
            y_contrast = y_contrast.contiguous().view(-1, 1)
            contrast_count = 1
            contrast_feature = X_contrast
        else:
            y_contrast = y_anchor
            contrast_count = n_view
            contrast_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)

        # (anchor_num × n_view) × (anchor_num × n_view)
        mask = torch.eq(y_anchor, y_contrast.T).float().to(self.device)
        # (anchor_num × n_view) × (anchor_num × n_view)
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).\
            scatter_(1, torch.arange(anchor_num * anchor_count).view(-1, 1).to(self.device), 0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-5)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        return loss

    def forward(self, feats, labels=None, predict=None, queue=None):
        labels = labels.float().clone()
        labels = torch.nn.functional.interpolate(labels, (feats.shape[2], feats.shape[3]), mode='nearest')
        predict = torch.nn.functional.interpolate(predict, (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])

        # feats: N×(HW)×C
        # labels: N×(HW)
        # predict: N×(HW)
        feats_, labels_ = self._anchor_sampling(feats, labels, predict)
        loss = self._contrastive(feats_, labels_, queue=queue)
        return loss
