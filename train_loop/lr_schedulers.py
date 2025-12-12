
import math
from train_loop.losses import opt_learning_rate

class CosineLRScheduler:
    def __init__(self, optimizer , 
                 total_steps,
                 base_lr=None,
                 min_lr=0.0,
                 warmup_lr=0.0,
                 warmup_steps=0,
                 steps_per_epoch=None,
                 epoch_wise=False,
                 ):
        
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_lr = warmup_lr
        self.min_lr = min_lr
        self.steps_per_epoch = steps_per_epoch
        self.total_iterations = total_steps
        self.warmup_steps = warmup_steps
        self.epoch_wise = epoch_wise
        if self.base_lr is None:
            self.base_lr = opt_learning_rate(optimizer)

    def step(self, it):
        
        if self.epoch_wise:
            epoch = it // self.steps_per_epoch
            it = epoch * self.steps_per_epoch
        
        # Warmup from warmup_lr to base_lr
        if it < self.warmup_steps:
            lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * it / self.warmup_steps
            return lr
        
        # Step 3: Cosine annealing after warmup
        # Progress through cosine schedule (0 to 1)
        progress = (it - self.warmup_steps) / (self.total_iterations - self.warmup_steps)
        
        # Cosine decay from self.base_lr to min_lr
        lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
        
        for group in self.optimizer.param_groups:
            group["lr"] = lr 
