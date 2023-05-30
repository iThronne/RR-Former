# Learning rate adjusting strategies.

from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR


class Ordinary:
    def __init__(self):
        pass

    def _ordinary_lr(self, epoch):
        return 1

    def get_scheduler(self, optimizer):
        scheduler = LambdaLR(optimizer, self._ordinary_lr, last_epoch=-1)
        return scheduler


class WarmUp:
    """
    Warm up strategy.
    """

    def __init__(self, warm_up_epochs, decay_rate):
        self.warm_up_epochs = warm_up_epochs
        self.decay_rate = decay_rate

    def _warm_up_lr(self, epoch):
        if epoch < self.warm_up_epochs:
            return (epoch + 1) / self.warm_up_epochs
        else:
            return self.decay_rate ** (epoch - self.warm_up_epochs)

    def get_scheduler(self, optimizer):
        scheduler = LambdaLR(optimizer, self._warm_up_lr, last_epoch=-1)
        return scheduler


class ExpDecay:
    """
    Exponential decay strategy.
    """

    def __init__(self, decay_epoch, decay_rate):
        self.decay_epoch = decay_epoch
        self.decay_rate = decay_rate

    def _exp_decay_lr(self, epoch):
        if epoch <= self.decay_epoch:
            return 1
        else:
            return self.decay_rate ** (epoch - self.decay_epoch)

    def get_scheduler(self, optimizer):
        scheduler = LambdaLR(optimizer, self._exp_decay_lr, last_epoch=-1)
        return scheduler


class CosineAnnealing:
    """
    CosineAnnealing strategy.
    """

    def __init__(self, cos_anneal_t_max):
        self.cos_anneal_t_max = cos_anneal_t_max

    def get_scheduler(self, optimizer):
        scheduler = CosineAnnealingLR(optimizer, T_max=self.cos_anneal_t_max, eta_min=0, last_epoch=-1)
        return scheduler


class SchedulerFactory:
    """
    Simple factory for producing learning rate scheduler
    """

    @staticmethod
    def get_scheduler(optimizer, scheduler_type, **kwargs):
        if scheduler_type == "none":
            scheduler = Ordinary().get_scheduler(optimizer)
        elif scheduler_type == "warm_up":
            scheduler = WarmUp(**kwargs).get_scheduler(optimizer)
        elif scheduler_type == "exp_decay":
            scheduler = ExpDecay(**kwargs).get_scheduler(optimizer)
        elif scheduler_type == "cos_anneal":
            scheduler = CosineAnnealing(**kwargs).get_scheduler(optimizer)
        else:
            raise RuntimeError(f"No such scheduler type: {scheduler_type}")

        return scheduler
