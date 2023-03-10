import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import torch

import wandb

import sys

import constants
import lib
from lib import callbacks
from lib import NotALightningTrainer
from lib.loggers import WandbLogger

import nomenclature

from lib.utils import load_model_by_name
from lib.arg_utils import define_args

print(":::: python " + ' '.join(sys.argv))

args = define_args(extra_args = [
    ('--runs', {'default': None, 'type': int, 'required': False}),
    ('--fraction', {'default': None, 'type': float, 'required': False}),
    ('--checkpoint', {'default': None, 'type': str, 'required': False}),
])
wandb.init(project = 'pose-transformers', group = args.group, entity="ucs")
wandb.config.update(vars(args))

dataset = nomenclature.DATASETS[args.dataset]
train_dataloader = nomenclature.DATASETS[args.dataset].train_dataloader(args)

architecture = nomenclature.MODELS[args.model](args)
architecture.to(lib.device)

if args.checkpoint is not None:
    state_dict = load_model_by_name(args.checkpoint)
    state_dict = {
        key.replace('module.', ''): value
        for key, value in state_dict.items()
    }

    missing_keys = architecture.load_state_dict(state_dict, strict = False)
    print(':::', missing_keys)
    architecture.eval()
    architecture.train(False)
    architecture.to(lib.device)

model = nomenclature.TRAINER[args.trainer](args, architecture)

if isinstance(args.evaluators, str):
    args.evaluators = [args.evaluators]

evaluators = [
    nomenclature.EVALUATORS[evaluator_name](args, architecture)
    for evaluator_name in args.evaluators
]

wandb_logger = WandbLogger()

checkpoint_callback_best = callbacks.ModelCheckpoint(
    name = ' 🔥 Best Checkpoint Overall 🔥',
    monitor = args.model_checkpoint['monitor_quantity'],
    dirpath = f'checkpoints/{args.group}:{args.name}/best/',
    save_weights_only = True,
    save_best_only = True,
    direction = args.model_checkpoint['direction'],
    filename=f'epoch={{epoch}}-{args.model_checkpoint["monitor_quantity"]}={{{args.model_checkpoint["monitor_quantity"]}:.4f}}.ckpt',
)

checkpoint_callback_last = callbacks.ModelCheckpoint(
    name = '🛠️ Last Checkpoint 🛠️',
    monitor = args.model_checkpoint['monitor_quantity'],
    dirpath = f'checkpoints/{args.group}:{args.name}/last/',
    save_weights_only = True,
    save_best_only = False,
    direction = args.model_checkpoint['direction'],
    filename=f'epoch={{epoch}}-{args.model_checkpoint["monitor_quantity"]}={{{args.model_checkpoint["monitor_quantity"]}:.4f}}.ckpt',
)

if bool(args.use_scheduler):
    scheduler_args = args.lr_scheduler

    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer = model.configure_optimizers(lr = scheduler_args.base_lr),
        cycle_momentum = False,
        base_lr = scheduler_args.base_lr,
        mode = scheduler_args.mode,
        step_size_up = len(train_dataloader) * scheduler_args.step_size_up, # per epoch
        step_size_down = len(train_dataloader) * scheduler_args.step_size_down, # per epoch
        max_lr = scheduler_args.max_lr
    )

    lr_callback = callbacks.LambdaCallback(
        on_batch_end = lambda: scheduler.step()
    )

    lr_logger = callbacks.LambdaCallback(
        on_batch_end = lambda: wandb_logger.log('lr', scheduler.get_last_lr()[0])
    )
    lr_callbacks = [lr_logger, lr_callback]
else:
    lr_callbacks = []

trainer = NotALightningTrainer(
    args = args,
    callbacks = [
        checkpoint_callback_last,
        checkpoint_callback_best,
    ] + lr_callbacks,
    logger=wandb_logger,
)

torch.backends.cudnn.benchmark = True
trainer.fit(
    model,
    train_dataloader,
    evaluators = evaluators
)
