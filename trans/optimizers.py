"""Optimizer classes and lr scheduler used in training."""

import argparse
from trans import register_component

import torch


@register_component('adam', 'optimizer')
class Adam(torch.optim.Adam):
    def __init__(self, params, dargs: dict):
        super().__init__(
            params,
            lr=dargs['lr'],
            betas=dargs['betas'],
            eps=dargs['eps'],
            weight_decay=dargs['weight_decay'],
            amsgrad=dargs['amsgrad']
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--betas", type=tuple, default=(0.9, 0.999))
        parser.add_argument("--eps", type=float, default=1e-08)
        parser.add_argument("--weight-decay", type=float, default=0)
        parser.add_argument("--amsgrad", type=bool, default=False)


@register_component('adadelta', 'optimizer')
class Adadelta(torch.optim.Adadelta):
    def __init__(self, params, dargs: dict):
        super().__init__(
            params,
            lr=dargs['lr'],
            rho=dargs['rho'],
            eps=dargs['eps'],
            weight_decay=dargs['weight_decay']
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--lr", type=float, default=1.0)
        parser.add_argument("--rho", type=float, default=0.9)
        parser.add_argument("--eps", type=float, default=1e-06)
        parser.add_argument("--weight-decay", type=float, default=0.)


@register_component('inv_sr', 'lr_scheduler')
class InvSRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: torch.optim, dargs: dict):
        self.warmup_steps = dargs['warmup_steps']
        lr = optimizer.param_groups[0]['lr']
        self.init_lr = dargs['warmup_init_lr'] if dargs['warmup_steps'] > 0 else lr
        self.final_lr = lr
        super().__init__(
            optimizer,
            last_epoch=dargs['last_epoch'],
            verbose=dargs['verbose']
        )

    def get_lr(self):
        if self.last_epoch == 0:
            return [self.init_lr for group in self.optimizer.param_groups]
        if self.last_epoch < self.warmup_steps:
            lr = self.init_lr + ((self.final_lr - self.init_lr) / self.warmup_steps) * self.last_epoch
        else:
            lr = self.final_lr * (self.warmup_steps ** 0.5 if self.warmup_steps > 0 else 1) * self.last_epoch**-0.5
        return [lr for group in self.optimizer.param_groups]

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--warmup-init-lr", type=float, default=0.00001)
        parser.add_argument("--warmup-steps", type=int, default=20)
        parser.add_argument("--last-epoch", type=int, default=-1)
        parser.add_argument("--verbose", type=bool, default=False)
