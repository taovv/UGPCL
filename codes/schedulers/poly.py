from torch.optim.lr_scheduler import _LRScheduler


class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters + 1  # avoid zero lr
        self.min_lr = min_lr
        self.last_epoch = last_epoch
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        '''
        factor = pow(1 - self.last_epoch / self.max_iters, self.power)
        return [(base_lr) * factor for base_lr in self.base_lrs]
        '''
        return [max(base_lr * (1.0 - self.last_epoch / self.max_iters) ** self.power, self.min_lr)
                for base_lr in self.base_lrs]

    def __str__(self):
        return f'PolyLR(' \
               f'\n\tpower: {self.power}' \
               f'\n\tmax_iters: {self.max_iters}' \
               f'\n\tmin_lr: {self.min_lr}' \
               f'\n\tlast_epoch: {self.last_epoch}\n)'
