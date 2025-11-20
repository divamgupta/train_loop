

class AKLRScheduler:
    def __init__(self, optimizer , total_iterations , muon_optimizer_name , warmup_ratio , warmdown_ratio, final_lr_frac):
        self.optimizer = optimizer
        self.muon_optimizer_name = muon_optimizer_name
        self.total_iterations = total_iterations
        self.warmup_ratio = warmup_ratio
        self.warmdown_ratio = warmdown_ratio
        self.final_lr_frac = final_lr_frac

    def step(self, it):

        assert self.optimizer.type == 'multi'

        num_iterations = self.total_iterations
        warmup_ratio = self.warmup_ratio
        warmdown_ratio = self.warmdown_ratio
        final_lr_frac = self.final_lr_frac


        def get_lr_multiplier(it):
            warmup_iters = round(warmup_ratio * num_iterations)
            warmdown_iters = round(warmdown_ratio * num_iterations)
            if it < warmup_iters:
                return (it + 1) / warmup_iters
            elif it <= num_iterations - warmdown_iters:
                return 1.0
            else:
                progress = (num_iterations - it) / warmdown_iters
                r =  progress * 1.0 + (1 - progress) * final_lr_frac
                r = max(r, final_lr_frac)
                return r

        # Momentum scheduler for Muon optimizer
        def get_muon_momentum(it):
            frac = min(it / 300, 1)
            momentum = (1 - frac) * 0.85 + frac * 0.95
            return momentum

        lrm = get_lr_multiplier(it)

        muon_momentum = get_muon_momentum(it)

        for opt_name in self.optimizer.optimizers:
            opt = self.optimizer.optimizers[opt_name]

            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * lrm

        for group in self.optimizer.optimizers[self.muon_optimizer_name].param_groups:
            group["momentum"] = muon_momentum

        print(f"Iteration {it}: LR multiplier={lrm:.6f}, Muon momentum={muon_momentum:.6f}")