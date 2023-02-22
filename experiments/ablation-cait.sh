#!/bin/bash

set -e
cd ..

# for debugging use `dryrun`, otherwise use `run`
WANDB_MODE=run

GROUP=ablation_final
ENV=adrian-exodus
EPOCHS=50
BATCH_SIZE=256

TRAIN_TAIL="--dataset casia --group $GROUP --batch_size $BATCH_SIZE --mode $WANDB_MODE --epochs $EPOCHS --env $ENV"
EVAL_TAIL="--group $GROUP --env $ENV --eval_config configs/evaluation/recognition.yaml --output_dir $GROUP"

#################
# CaiT
python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-64-2 $TRAIN_TAIL --model_args.patch_h 64 --model_args.patch_w 2
python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-64-4 $TRAIN_TAIL --model_args.patch_h 64 --model_args.patch_w 4
python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-64-8 $TRAIN_TAIL --model_args.patch_h 64 --model_args.patch_w 8
python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-64-16 $TRAIN_TAIL --model_args.patch_h 64 --model_args.patch_w 16
python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-64-32 $TRAIN_TAIL --model_args.patch_h 64 --model_args.patch_w 32

python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-32-2 $TRAIN_TAIL --model_args.patch_h 32 --model_args.patch_w 2
python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-32-4 $TRAIN_TAIL --model_args.patch_h 32 --model_args.patch_w 4
python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-32-8 $TRAIN_TAIL --model_args.patch_h 32 --model_args.patch_w 8
python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-32-16 $TRAIN_TAIL --model_args.patch_h 32 --model_args.patch_w 16
python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-32-32 $TRAIN_TAIL --model_args.patch_h 32 --model_args.patch_w 32
python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-32-64 $TRAIN_TAIL --model_args.patch_h 32 --model_args.patch_w 64

python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-16-2 $TRAIN_TAIL --model_args.patch_h 16 --model_args.patch_w 2
python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-16-4 $TRAIN_TAIL --model_args.patch_h 16 --model_args.patch_w 4
python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-16-8 $TRAIN_TAIL --model_args.patch_h 16 --model_args.patch_w 8
python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-16-16 $TRAIN_TAIL --model_args.patch_h 16 --model_args.patch_w 16
python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-16-32 $TRAIN_TAIL --model_args.patch_h 16 --model_args.patch_w 32
python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-16-64 $TRAIN_TAIL --model_args.patch_h 16 --model_args.patch_w 64

python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-8-2 $TRAIN_TAIL --model_args.patch_h 8 --model_args.patch_w 2
python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-8-4 $TRAIN_TAIL --model_args.patch_h 8 --model_args.patch_w 4
python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-8-8 $TRAIN_TAIL --model_args.patch_h 8 --model_args.patch_w 8
python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-8-16 $TRAIN_TAIL --model_args.patch_h 8 --model_args.patch_w 16
python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-8-32 $TRAIN_TAIL --model_args.patch_h 8 --model_args.patch_w 32

python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-2-4 $TRAIN_TAIL --model_args.patch_h 2 --model_args.patch_w 4
python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-2-8 $TRAIN_TAIL --model_args.patch_h 2 --model_args.patch_w 8
python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-2-16 $TRAIN_TAIL --model_args.patch_h 2 --model_args.patch_w 16
python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-2-32 $TRAIN_TAIL --model_args.patch_h 2 --model_args.patch_w 32
python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-2-64 $TRAIN_TAIL --model_args.patch_h 2 --model_args.patch_w 64

python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-8-64 $TRAIN_TAIL --model_args.patch_h 8 --model_args.patch_w 64

python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-4-2 $TRAIN_TAIL --model_args.patch_h 4 --model_args.patch_w 2
python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-4-4 $TRAIN_TAIL --model_args.patch_h 4 --model_args.patch_w 4
python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-4-8 $TRAIN_TAIL --model_args.patch_h 4 --model_args.patch_w 8
python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-4-16 $TRAIN_TAIL --model_args.patch_h 4 --model_args.patch_w 16
python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-4-32 $TRAIN_TAIL --model_args.patch_h 4 --model_args.patch_w 32
python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-4-64 $TRAIN_TAIL --model_args.patch_h 4 --model_args.patch_w 64


python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-4-1 $TRAIN_TAIL --model_args.patch_h 4 --model_args.patch_w 1
python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-8-1 $TRAIN_TAIL --model_args.patch_h 8 --model_args.patch_w 1
python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-16-1 $TRAIN_TAIL --model_args.patch_h 16 --model_args.patch_w 1
python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-32-1 $TRAIN_TAIL --model_args.patch_h 32 --model_args.patch_w 1


python main.py --config_file configs/model_configs/cait.yaml --name cait-patch-1-64 $TRAIN_TAIL --model_args.patch_h 1 --model_args.patch_w 64


############################################################################3
############################################################################3
############################################################################3
############################################################################3
python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-64-2 $EVAL_TAIL --model_args.patch_h 64 --model_args.patch_w 2
python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-64-4 $EVAL_TAIL --model_args.patch_h 64 --model_args.patch_w 4
python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-64-8 $EVAL_TAIL --model_args.patch_h 64 --model_args.patch_w 8
python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-64-16 $EVAL_TAIL --model_args.patch_h 64 --model_args.patch_w 16
python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-64-32 $EVAL_TAIL --model_args.patch_h 64 --model_args.patch_w 32

python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-32-2 $EVAL_TAIL --model_args.patch_h 32 --model_args.patch_w 2
python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-32-4 $EVAL_TAIL --model_args.patch_h 32 --model_args.patch_w 4
python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-32-8 $EVAL_TAIL --model_args.patch_h 32 --model_args.patch_w 8
python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-32-16 $EVAL_TAIL --model_args.patch_h 32 --model_args.patch_w 16
python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-32-32 $EVAL_TAIL --model_args.patch_h 32 --model_args.patch_w 32
python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-32-64 $EVAL_TAIL --model_args.patch_h 32 --model_args.patch_w 64

python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-16-2 $EVAL_TAIL --model_args.patch_h 16 --model_args.patch_w 2
python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-16-4 $EVAL_TAIL --model_args.patch_h 16 --model_args.patch_w 4
python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-16-8 $EVAL_TAIL --model_args.patch_h 16 --model_args.patch_w 8
python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-16-16 $EVAL_TAIL --model_args.patch_h 16 --model_args.patch_w 16
python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-16-32 $EVAL_TAIL --model_args.patch_h 16 --model_args.patch_w 32
python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-16-64 $EVAL_TAIL --model_args.patch_h 16 --model_args.patch_w 64

python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-8-2 $EVAL_TAIL --model_args.patch_h 8 --model_args.patch_w 2
python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-8-4 $EVAL_TAIL --model_args.patch_h 8 --model_args.patch_w 4
python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-8-8 $EVAL_TAIL --model_args.patch_h 8 --model_args.patch_w 8
python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-8-16 $EVAL_TAIL --model_args.patch_h 8 --model_args.patch_w 16
python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-8-32 $EVAL_TAIL --model_args.patch_h 8 --model_args.patch_w 32

python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-2-4 $EVAL_TAIL --model_args.patch_h 2 --model_args.patch_w 4
python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-2-8 $EVAL_TAIL --model_args.patch_h 2 --model_args.patch_w 8
python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-2-16 $EVAL_TAIL --model_args.patch_h 2 --model_args.patch_w 16
python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-2-32 $EVAL_TAIL --model_args.patch_h 2 --model_args.patch_w 32
python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-2-64 $EVAL_TAIL --model_args.patch_h 2 --model_args.patch_w 64

python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-8-64 $EVAL_TAIL --model_args.patch_h 8 --model_args.patch_w 64

python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-4-2 $EVAL_TAIL --model_args.patch_h 4 --model_args.patch_w 2
python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-4-4 $EVAL_TAIL --model_args.patch_h 4 --model_args.patch_w 4
python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-4-8 $EVAL_TAIL --model_args.patch_h 4 --model_args.patch_w 8
python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-4-16 $EVAL_TAIL --model_args.patch_h 4 --model_args.patch_w 16
python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-4-32 $EVAL_TAIL --model_args.patch_h 4 --model_args.patch_w 32
python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-4-64 $EVAL_TAIL --model_args.patch_h 4 --model_args.patch_w 64


python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-4-1 $EVAL_TAIL --model_args.patch_h 4 --model_args.patch_w 1
python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-8-1 $EVAL_TAIL --model_args.patch_h 8 --model_args.patch_w 1
python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-16-1 $EVAL_TAIL --model_args.patch_h 16 --model_args.patch_w 1
python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-32-1 $EVAL_TAIL --model_args.patch_h 32 --model_args.patch_w 1


python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-patch-1-64 $EVAL_TAIL --model_args.patch_h 1 --model_args.patch_w 64
