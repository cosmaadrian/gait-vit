#!/bin/bash

set -e
cd ..

# for debugging use `dryrun`, otherwise use `run`
WANDB_MODE=run
ENV=adrian-exodus

GROUP=$1
ARCH=$2

EPOCHS=50
BATCH_SIZE=512

CONFIG=configs/model_configs/$ARCH.yaml

if [ -n "$1" ]
then
    GROUP=$1
fi


# FROM SCRATCH RECOGNITION CASIA
python main.py --config_file $CONFIG --name $ARCH-scratch-casia-1 --dataset casia --runs 1 --group $GROUP --mode $WANDB_MODE --batch_size $BATCH_SIZE --epochs $EPOCHS --env $ENV
python main.py --config_file $CONFIG --name $ARCH-scratch-casia-2 --dataset casia --runs 2 --group $GROUP --mode $WANDB_MODE --batch_size $BATCH_SIZE --epochs $EPOCHS --env $ENV
python main.py --config_file $CONFIG --name $ARCH-scratch-casia-3 --dataset casia --runs 3 --group $GROUP --mode $WANDB_MODE --batch_size $BATCH_SIZE --epochs $EPOCHS --env $ENV
python main.py --config_file $CONFIG --name $ARCH-scratch-casia-5 --dataset casia --runs 5 --group $GROUP --mode $WANDB_MODE --batch_size $BATCH_SIZE --epochs $EPOCHS --env $ENV
python main.py --config_file $CONFIG --name $ARCH-scratch-casia-7 --dataset casia --runs 7 --group $GROUP --mode $WANDB_MODE --batch_size $BATCH_SIZE --epochs $EPOCHS --env $ENV
python main.py --config_file $CONFIG --name $ARCH-scratch-casia-9 --dataset casia --runs 9 --group $GROUP --mode $WANDB_MODE --batch_size $BATCH_SIZE --epochs $EPOCHS --env $ENV
python main.py --config_file $CONFIG --name $ARCH-scratch-casia-10 --dataset casia --runs 10 --group $GROUP --mode $WANDB_MODE --batch_size $BATCH_SIZE --epochs $EPOCHS --env $ENV

# FROM SCRATCH RECOGNITION FVG
python main.py --config_file $CONFIG --name $ARCH-scratch-fvg-0.1 --dataset fvg --fraction 0.1 --group $GROUP --mode $WANDB_MODE --batch_size $BATCH_SIZE --epochs $EPOCHS --env $ENV --evaluators fvg-recognition --model_checkpoint.monitor_quantity FVGRecognitionEvaluator_ALL
python main.py --config_file $CONFIG --name $ARCH-scratch-fvg-0.2 --dataset fvg --fraction 0.2 --group $GROUP --mode $WANDB_MODE --batch_size $BATCH_SIZE --epochs $EPOCHS --env $ENV --evaluators fvg-recognition --model_checkpoint.monitor_quantity FVGRecognitionEvaluator_ALL
python main.py --config_file $CONFIG --name $ARCH-scratch-fvg-0.3 --dataset fvg --fraction 0.3 --group $GROUP --mode $WANDB_MODE --batch_size $BATCH_SIZE --epochs $EPOCHS --env $ENV --evaluators fvg-recognition --model_checkpoint.monitor_quantity FVGRecognitionEvaluator_ALL
python main.py --config_file $CONFIG --name $ARCH-scratch-fvg-0.5 --dataset fvg --fraction 0.5 --group $GROUP --mode $WANDB_MODE --batch_size $BATCH_SIZE --epochs $EPOCHS --env $ENV --evaluators fvg-recognition --model_checkpoint.monitor_quantity FVGRecognitionEvaluator_ALL
python main.py --config_file $CONFIG --name $ARCH-scratch-fvg-0.7 --dataset fvg --fraction 0.7 --group $GROUP --mode $WANDB_MODE --batch_size $BATCH_SIZE --epochs $EPOCHS --env $ENV --evaluators fvg-recognition --model_checkpoint.monitor_quantity FVGRecognitionEvaluator_ALL
python main.py --config_file $CONFIG --name $ARCH-scratch-fvg-0.9 --dataset fvg --fraction 0.9 --group $GROUP --mode $WANDB_MODE --batch_size $BATCH_SIZE --epochs $EPOCHS --env $ENV --evaluators fvg-recognition --model_checkpoint.monitor_quantity FVGRecognitionEvaluator_ALL
python main.py --config_file $CONFIG --name $ARCH-scratch-fvg-1 --dataset fvg --fraction 1.0 --group $GROUP --mode $WANDB_MODE --batch_size $BATCH_SIZE --epochs $EPOCHS --env $ENV --evaluators fvg-recognition --model_checkpoint.monitor_quantity FVGRecognitionEvaluator_ALL
