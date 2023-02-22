#!/bin/bash

set -e
cd ../

# for debugging use `dryrun`, otherwise use `run`
WANDB_MODE=run
ENV=adrian-exodus

GROUP=$1
ARCH=$2
DATASET=$3

EPOCHS=50
BATCH_SIZE=512

CONFIG=configs/model_configs/$ARCH.yaml

CHECKPOINT="$GROUP:$ARCH-$DATASET"

#############################################################################################
#############################################################################################
#############################################################################################
python main.py --config_file configs/model_configs/$ARCH.yaml --name $ARCH-$DATASET --dataset $DATASET --group $GROUP --mode $WANDB_MODE --batch_size $BATCH_SIZE --epochs 200 --env $ENV
#############################################################################################
#############################################################################################
#############################################################################################

############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################

# FINE TUNE CASIA
python main.py --config_file $CONFIG --name $ARCH-$DATASET-tuned-casia-1 --dataset casia --runs 1 --group $GROUP --mode $WANDB_MODE --batch_size $BATCH_SIZE --epochs $EPOCHS --env $ENV --trainer fine-tuner --checkpoint $CHECKPOINT
python main.py --config_file $CONFIG --name $ARCH-$DATASET-tuned-casia-2 --dataset casia --runs 2 --group $GROUP --mode $WANDB_MODE --batch_size $BATCH_SIZE --epochs $EPOCHS --env $ENV --trainer fine-tuner --checkpoint $CHECKPOINT
python main.py --config_file $CONFIG --name $ARCH-$DATASET-tuned-casia-3 --dataset casia --runs 3 --group $GROUP --mode $WANDB_MODE --batch_size $BATCH_SIZE --epochs $EPOCHS --env $ENV --trainer fine-tuner --checkpoint $CHECKPOINT
python main.py --config_file $CONFIG --name $ARCH-$DATASET-tuned-casia-5 --dataset casia --runs 5 --group $GROUP --mode $WANDB_MODE --batch_size $BATCH_SIZE --epochs $EPOCHS --env $ENV --trainer fine-tuner --checkpoint $CHECKPOINT
python main.py --config_file $CONFIG --name $ARCH-$DATASET-tuned-casia-7 --dataset casia --runs 7 --group $GROUP --mode $WANDB_MODE --batch_size $BATCH_SIZE --epochs $EPOCHS --env $ENV --trainer fine-tuner --checkpoint $CHECKPOINT
python main.py --config_file $CONFIG --name $ARCH-$DATASET-tuned-casia-9 --dataset casia --runs 9 --group $GROUP --mode $WANDB_MODE --batch_size $BATCH_SIZE --epochs $EPOCHS --env $ENV --trainer fine-tuner --checkpoint $CHECKPOINT
python main.py --config_file $CONFIG --name $ARCH-$DATASET-tuned-casia-10 --dataset casia --runs 10 --group $GROUP --mode $WANDB_MODE --batch_size $BATCH_SIZE --epochs $EPOCHS --env $ENV --trainer fine-tuner --checkpoint $CHECKPOINT

# FINE TUNE FVG
python main.py --config_file $CONFIG --name $ARCH-$DATASET-tuned-fvg-0.1 --dataset fvg --fraction 0.1 --group $GROUP --mode $WANDB_MODE --batch_size $BATCH_SIZE --epochs $EPOCHS --env $ENV --trainer fine-tuner --checkpoint $CHECKPOINT --evaluators fvg-recognition --model_checkpoint.monitor_quantity FVGRecognitionEvaluator_ALL
python main.py --config_file $CONFIG --name $ARCH-$DATASET-tuned-fvg-0.2 --dataset fvg --fraction 0.2 --group $GROUP --mode $WANDB_MODE --batch_size $BATCH_SIZE --epochs $EPOCHS --env $ENV --trainer fine-tuner --checkpoint $CHECKPOINT --evaluators fvg-recognition --model_checkpoint.monitor_quantity FVGRecognitionEvaluator_ALL
python main.py --config_file $CONFIG --name $ARCH-$DATASET-tuned-fvg-0.3 --dataset fvg --fraction 0.3 --group $GROUP --mode $WANDB_MODE --batch_size $BATCH_SIZE --epochs $EPOCHS --env $ENV --trainer fine-tuner --checkpoint $CHECKPOINT --evaluators fvg-recognition --model_checkpoint.monitor_quantity FVGRecognitionEvaluator_ALL
python main.py --config_file $CONFIG --name $ARCH-$DATASET-tuned-fvg-0.5 --dataset fvg --fraction 0.5 --group $GROUP --mode $WANDB_MODE --batch_size $BATCH_SIZE --epochs $EPOCHS --env $ENV --trainer fine-tuner --checkpoint $CHECKPOINT --evaluators fvg-recognition --model_checkpoint.monitor_quantity FVGRecognitionEvaluator_ALL
python main.py --config_file $CONFIG --name $ARCH-$DATASET-tuned-fvg-0.7 --dataset fvg --fraction 0.7 --group $GROUP --mode $WANDB_MODE --batch_size $BATCH_SIZE --epochs $EPOCHS --env $ENV --trainer fine-tuner --checkpoint $CHECKPOINT --evaluators fvg-recognition --model_checkpoint.monitor_quantity FVGRecognitionEvaluator_ALL
python main.py --config_file $CONFIG --name $ARCH-$DATASET-tuned-fvg-0.9 --dataset fvg --fraction 0.9 --group $GROUP --mode $WANDB_MODE --batch_size $BATCH_SIZE --epochs $EPOCHS --env $ENV --trainer fine-tuner --checkpoint $CHECKPOINT --evaluators fvg-recognition --model_checkpoint.monitor_quantity FVGRecognitionEvaluator_ALL
python main.py --config_file $CONFIG --name $ARCH-$DATASET-tuned-fvg-1 --dataset fvg --fraction 1.0 --group $GROUP --mode $WANDB_MODE --batch_size $BATCH_SIZE --epochs $EPOCHS --env $ENV --trainer fine-tuner --checkpoint $CHECKPOINT --evaluators fvg-recognition --model_checkpoint.monitor_quantity FVGRecognitionEvaluator_ALL
