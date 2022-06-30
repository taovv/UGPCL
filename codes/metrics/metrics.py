"""
preds.shape: (N,1,H,W) || (N,1,H,W,D) || (N,H,W) || (N,H,W,D)
labels.shape: (N,1,H,W) || (N,1,H,W,D) || (N,H,W) || (N,H,W,D)
"""
import torch
from medpy import metric


class Dice:

    def __init__(self, name='Dice', class_indexs=[1], class_names=['xx']) -> None:
        super().__init__()
        self.name = name
        self.class_indexs = class_indexs
        self.class_names = class_names

    def __call__(self, preds, labels):
        res = {}
        for class_index, class_name in zip(self.class_indexs, self.class_names):
            preds_ = (preds == class_index).to(torch.int)
            labels_ = (labels == class_index).to(torch.int)
            intersection = (preds_ * labels_).sum()
            try:
                res[class_name] = (2 * intersection) / (preds_.sum() + labels_.sum()).item()
            except ZeroDivisionError:
                res[class_name] = 1.0
            # res[class_name] = metric.dc(preds_.cpu().numpy(), labels_.cpu().numpy())
        return res


class Jaccard:
    def __init__(self, name='Jaccard', class_indexs=[1], class_names=['xx']) -> None:
        super().__init__()
        self.name = name
        self.class_indexs = class_indexs
        self.class_names = class_names

    def __call__(self, preds, labels):
        res = {}
        for class_index, class_name in zip(self.class_indexs, self.class_names):
            preds_ = (preds == class_index).to(torch.int)
            labels_ = (labels == class_index).to(torch.int)
            intersection = (preds_ * labels_).sum()
            union = ((preds_ + labels_) != 0).sum()
            res[class_name] = intersection / union
            try:
                res[class_name] = intersection / union
            except ZeroDivisionError:
                res[class_name] = 1.0
            # res[class_name] = metric.jc(preds_.cpu().numpy(), labels_.cpu().numpy())
        return res


class HD95:
    """
    95th percentile of the Hausdorff Distance.
    """

    def __init__(self, name='95HD', class_indexs=[1], class_names=['xx']) -> None:
        super().__init__()
        self.name = name
        self.class_indexs = class_indexs
        self.class_names = class_names

    def __call__(self, preds, labels):
        if preds.size(1) == 1:
            preds = preds.squeeze(1)
        if labels.size(1) == 1:
            labels = labels.squeeze(1)
        res = {}
        for class_index, class_name in zip(self.class_indexs, self.class_names):
            res[class_name] = 0.
            for i in range(preds.size(0)):
                preds_ = (preds[i] == class_index).to(torch.int)
                labels_ = (labels[i] == class_index).to(torch.int)
                if preds_.sum() == 0.:
                    preds_ = (preds_ == 0).to(torch.int)
                res[class_name] += torch.tensor(metric.hd95(preds_.cpu().numpy(), labels_.cpu().numpy()))
            res[class_name] /= preds.size(0)
        return res


class ASD:
    """
    Average surface distance.
    """

    def __init__(self, name='ASD', class_indexs=[1], class_names=['xx']) -> None:
        super().__init__()
        self.name = name
        self.class_indexs = class_indexs
        self.class_names = class_names

    def __call__(self, preds, labels):
        if preds.size(1) == 1:
            preds = preds.squeeze(1)
        if labels.size(1) == 1:
            labels = labels.squeeze(1)
        res = {}
        for class_index, class_name in zip(self.class_indexs, self.class_names):
            res[class_name] = 0.
            for i in range(preds.size(0)):
                preds_ = (preds[i] == class_index).to(torch.int)
                labels_ = (labels[i] == class_index).to(torch.int)
                if preds_.sum() == 0.:
                    preds_ = (preds_ == 0).to(torch.int)
                res[class_name] += torch.tensor(metric.asd(preds_.cpu().numpy(), labels_.cpu().numpy()))
            res[class_name] /= preds.size(0)
        return res