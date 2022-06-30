import itertools
import numpy as np
from torch.utils.data.sampler import Sampler


class TwoStreamBatchSampler(Sampler):
    """
    Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.primary_batch_size = batch_size - secondary_batch_size
        self.secondary_batch_size = secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        """
        生成每个batch的数据采样索引：分别从labeled indices和unlabeled indices中选取
        每次调用索引序列都会被打乱，确保随机采样
        Returns: （labeled index1,labeled index2,...,unlabeled index1,unlabeled index2,...）

        """
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)  # 随机排列序列


def iterate_eternally(indices):
    """
    异步不断生成随机打乱的indices序列  并由itertools.chain连接后返回
    Args:
        indices:

    Returns:

    """
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    """
    Collect data into fixed-length chunks or blocks
    eg: grouper('ABCDEFG', 3) --> ABC DEF
    Args:
        iterable:
        n:

    Returns:

    """
    args = [iter(iterable)] * n
    return zip(*args)
