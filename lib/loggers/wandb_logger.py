import wandb

class WandbLogger(object):
    def __init__(self):
        self.on_step_metrics = dict()
        self.trainer = None

        self.metrics = dict()

    def watch(self, model):
        wandb.watch(model)

    def log_dict(self, log_dict, on_step = True, force_log = False):
        for key, value in log_dict.items():
            self.metrics[key] = value
            if on_step:
                self.on_step_metrics[key] = value

        log_every_filter = (self.trainer.global_step % self.trainer.args.log_every == 0)
        accumulation_filter = ((self.trainer.local_step + 1) % self.trainer.args.accumulation_steps == 0)

        if force_log or (log_every_filter and accumulation_filter):
            if wandb.run is not None:
                wandb.log(log_dict, step = self.trainer.global_step)

    def log(self, key, value, on_step = True, force_log = False):
        self.metrics[key] = value

        if on_step:
            self.on_step_metrics[key] = value

        log_every_filter = (self.trainer.global_step % self.trainer.args.log_every == 0)
        accumulation_filter = ((self.trainer.local_step + 1) % self.trainer.args.accumulation_steps == 0)

        if force_log or (log_every_filter and accumulation_filter):
            if wandb.run is not None:
                wandb.log({key: value}, step = self.trainer.global_step)

