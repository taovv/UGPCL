import torch
from abc import abstractmethod
from torch.nn import (BCELoss, BCEWithLogitsLoss, CrossEntropyLoss)


class BaseWeightLoss:
    def __init__(self, name='loss', weight=1.) -> None:
        super().__init__()
        self.name = name
        self.weight = weight

    @abstractmethod
    def _cal_loss(self, preds, targets, **kwargs):
        pass

    def __call__(self, preds, targets, **kwargs):
        return self._cal_loss(preds, targets, **kwargs) * self.weight


class BCELoss_(BaseWeightLoss):
    def __init__(self, name='loss_bce', weight=1., **kwargs) -> None:
        super().__init__(name, weight)
        self.loss = BCELoss(**kwargs)

    def _cal_loss(self, preds, targets, **kwargs):
        return self.loss(preds, targets)


class BCEWithLogitsLoss_(BaseWeightLoss):
    def __init__(self, name='loss_bce', weight=1., **kwargs) -> None:
        super().__init__(name, weight)
        self.loss = BCEWithLogitsLoss(**kwargs)

    def _cal_loss(self, preds, targets, **kwargs):
        return self.loss(preds, targets)


class CrossEntropyLoss_(BaseWeightLoss):
    def __init__(self, name='loss_ce', weight=1., **kwargs) -> None:
        super().__init__(name, weight)
        self.loss = CrossEntropyLoss(**kwargs)

    def _cal_loss(self, preds, targets, **kwargs):
        targets = targets.to(torch.long).squeeze(1)
        return self.loss(preds, targets)


class BinaryDiceLoss_(BaseWeightLoss):
    def __init__(self, name='loss_dice', weight=1., smooth=1e-5, softmax=True, **kwargs) -> None:
        super().__init__(name, weight)
        self.smooth = smooth
        self.softmax = softmax

    def _cal_loss(self, preds, targets, **kwargs):
        assert preds.shape[0] == targets.shape[0]
        if self.softmax:
            preds = torch.argmax(torch.softmax(preds, dim=1), dim=1, keepdim=True).to(torch.float32)
        intersect = torch.sum(torch.mul(preds, targets))
        loss = 1 - (2 * intersect + self.smooth) / (torch.sum(preds.pow(2)) + torch.sum(targets.pow(2)) + self.smooth)
        return loss


class DiceLoss_(BaseWeightLoss):
    def __init__(self, name='loss_dice', weight=1., smooth=1e-5, n_classes=2, class_weight=None, softmax=True,
                 **kwargs):
        super().__init__(name, weight)
        self.n_classes = n_classes
        self.smooth = smooth
        self.class_weight = [1.] * self.n_classes if class_weight is None else class_weight
        self.softmax = softmax

    def _one_hot_encoder(self, targets):
        target_list = []
        for _ in range(self.n_classes):
            temp_prob = targets == _ * torch.ones_like(targets)
            target_list.append(temp_prob)
        output_tensor = torch.cat(target_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, pred, target):
        assert pred.shape[0] == target.shape[0]
        intersect = torch.sum(torch.mul(pred, target))
        loss = 1 - (2 * intersect + self.smooth) / (torch.sum(pred.pow(2)) + torch.sum(target.pow(2)) + self.smooth)
        return loss

    def _cal_loss(self, preds, targets, **kwargs):
        if self.softmax:
            preds = torch.softmax(preds, dim=1)
        targets = self._one_hot_encoder(targets)
        assert preds.size() == targets.size(), 'pred & target shape do not match'
        loss = 0.0
        for _ in range(self.n_classes):
            dice = self._dice_loss(preds[:, _], targets[:, _])
            loss += dice * self.class_weight[_]
        return loss / self.n_classes
