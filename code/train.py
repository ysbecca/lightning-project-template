import yaml
import sys
import os
import numpy as np
import math

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append(os.path.abspath('..'))

from pytorch_lightning.loggers import TensorBoardLogger

import argparse
import torch

from dataset_models.data_modules import *
from dataset_models.datasets import *

from network_models.densenet import *

from lit_models.litsgmcmc import *


from utils import *

from global_config import *


# Set up training arguments and parse
parser = argparse.ArgumentParser(description='training network')

parser.add_argument(
    '--task_id', type=int, default=-1,
    help='task id; if present, use this to look for config yaml file that overrides all other args')

parser.add_argument(
    '--dataset_code', type=str,
    help='the dataset string code')
parser.add_argument(
    '--model_code', type=str, default=DEFAULT_MODEL,
    help='model code for model architecture to use')

parser.add_argument(
    '--optim_code', type=str, default=DEFAULT_OPTIM,
    help='optimizer code for training')

parser.add_argument(
    '--training_code', type=str, default=DEFAULT_TRAINING_CODE,
    help='training type code for deterministic / VI / mc-dropout method of training')
parser.add_argument(
    '--max_epochs', type=int, default=DEFAULT_MAX_EPOCH,
    help='max umber of epochs to train for')
parser.add_argument(
    '--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
    help='training batch size')
parser.add_argument(
    '--patience', type=int, default=DEFAULT_PATIENCE,
    help='lr scheduler patience, used only if use_scheduler True')
parser.add_argument(
    '--lr', type=float, default=DEFAULT_LR,
    help='initial learning rate')

parser.add_argument(
    '--num_gpus', type=int, default=GPU_COUNT,
    help='number of gpus')
parser.add_argument(
    '--dev_run', type=bool, default=False,
    help='runs a dev testing run on limited num of batches')
parser.add_argument(
    '--use_scheduler', type=bool, default=True,
    help='use a learning rate scheduler during training')

parser.add_argument(
    '--ckpt', type=str, default=None,
    help='start training from this ckpt')
parser.add_argument(
    '--M', type=int, default=DEFAULT_M,
    help='uncertainty forward MC samples')



args = parser.parse_args()

if args.task_id > -1:
    # override with yaml configs if necessary
    args = override_from_config(args)


print("=======================================")
print("         TRAINING PARAMS               ")
print("=======================================")
for arg in vars(args):
    print(F"{arg:>20} {getattr(args, arg)}")


assert args.dataset_code in DATASET_CODES
assert args.model_code in MODEL_CODES
assert args.training_code in TRAINING_CODES
assert args.optim_code in OPTIM_CODES



########################################################################
# 
#           Logger
# 
########################################################################

model_desc = get_model_desc(args, args.beta, full=True)
print(model_desc)
logger = TensorBoardLogger(TB_LOGS_PATH, name=model_desc)

########################################################################
# 
#           Dataset
# 
########################################################################

if args.dataset_code == CIFAR_CODE:
    data_module = DataModuleCIFAR(batch_size=args.batch_size)
else:
    pass

########################################################################
# 
#           Setup Lightning module
# 
########################################################################

ckpt_path = None
if args.ckpt:
    ckpt_path = f"{TB_LOGS_PATH}"
    if args.ckpt[0] == "_":
        # load from code
        ckpt_path += FIXED_CKPTS[args.ckpt]
    else:
        ckpt_path += args.ckpt


print(f'Checkpoint: {ckpt_path}')


if args.training_code == SG_CODE:
    model = LitSGMCMC(args)

else:
    pass

########################################################################
# 
#           Training
# 
########################################################################

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    filename=str(args.task_id) + '-{epoch}-{val_loss:.4f}-{val_acc:.4f}',
    save_last=True, # for resuming training
    save_top_k=5,
)


lr_monitor = LearningRateMonitor(logging_interval='epoch')

val_steps = 1 if IS_LOCAL else -1

# init trainer
trainer = pl.Trainer(
	gpus=args.num_gpus,
	max_epochs=args.max_epochs,
    logger=logger,
    checkpoint_callback=True,
    callbacks=[checkpoint_callback, lr_monitor],
    fast_dev_run=args.dev_run,
    resume_from_checkpoint=ckpt_path,
    num_sanity_val_steps=val_steps, # run complete pass through val set
)

trainer.fit(model, data_module)

print(f"Model desc {model_desc:>20}")




