#!/bin/bash

set -e
cd ..

# for debugging use `dryrun`, otherwise use `run`
ENV=adrian-exodus

BATCH_SIZE=512

GROUP=$1
ARCH=$2
DATASET=$3

CONFIG=configs/model_configs/$ARCH.yaml
OUTPUT_DIR=$GROUP

#############################################################################################
#############################################################################################
#############################################################################################
python evaluate.py --config_file $CONFIG --name $ARCH-$DATASET --group $GROUP --batch_size $BATCH_SIZE --env $ENV --eval_config configs/evaluation/recognition.yaml --output_dir $OUTPUT_DIR
#############################################################################################
#############################################################################################
#############################################################################################

# FROM SCRATCH RECOGNITION CASIA
python evaluate.py --config_file $CONFIG --name $ARCH-scratch-casia-1 --group $GROUP --batch_size $BATCH_SIZE --env $ENV --eval_config configs/evaluation/recognition.yaml --output_dir $OUTPUT_DIR
python evaluate.py --config_file $CONFIG --name $ARCH-scratch-casia-2 --group $GROUP --batch_size $BATCH_SIZE --env $ENV --eval_config configs/evaluation/recognition.yaml --output_dir $OUTPUT_DIR
python evaluate.py --config_file $CONFIG --name $ARCH-scratch-casia-3 --group $GROUP --batch_size $BATCH_SIZE --env $ENV --eval_config configs/evaluation/recognition.yaml --output_dir $OUTPUT_DIR
python evaluate.py --config_file $CONFIG --name $ARCH-scratch-casia-5 --group $GROUP --batch_size $BATCH_SIZE --env $ENV --eval_config configs/evaluation/recognition.yaml --output_dir $OUTPUT_DIR
python evaluate.py --config_file $CONFIG --name $ARCH-scratch-casia-7 --group $GROUP --batch_size $BATCH_SIZE --env $ENV --eval_config configs/evaluation/recognition.yaml --output_dir $OUTPUT_DIR
python evaluate.py --config_file $CONFIG --name $ARCH-scratch-casia-9 --group $GROUP --batch_size $BATCH_SIZE --env $ENV --eval_config configs/evaluation/recognition.yaml --output_dir $OUTPUT_DIR
python evaluate.py --config_file $CONFIG --name $ARCH-scratch-casia-10 --group $GROUP --batch_size $BATCH_SIZE --env $ENV --eval_config configs/evaluation/recognition.yaml --output_dir $OUTPUT_DIR

# FROM SCRATCH RECOGNITION FVG
python evaluate.py --config_file $CONFIG --name $ARCH-scratch-fvg-0.1 --group $GROUP --batch_size $BATCH_SIZE --env $ENV --eval_config configs/evaluation/recognition.yaml --output_dir $OUTPUT_DIR
python evaluate.py --config_file $CONFIG --name $ARCH-scratch-fvg-0.2 --group $GROUP --batch_size $BATCH_SIZE --env $ENV --eval_config configs/evaluation/recognition.yaml --output_dir $OUTPUT_DIR
python evaluate.py --config_file $CONFIG --name $ARCH-scratch-fvg-0.3 --group $GROUP --batch_size $BATCH_SIZE --env $ENV --eval_config configs/evaluation/recognition.yaml --output_dir $OUTPUT_DIR
python evaluate.py --config_file $CONFIG --name $ARCH-scratch-fvg-0.5 --group $GROUP --batch_size $BATCH_SIZE --env $ENV --eval_config configs/evaluation/recognition.yaml --output_dir $OUTPUT_DIR
python evaluate.py --config_file $CONFIG --name $ARCH-scratch-fvg-0.7 --group $GROUP --batch_size $BATCH_SIZE --env $ENV --eval_config configs/evaluation/recognition.yaml --output_dir $OUTPUT_DIR
python evaluate.py --config_file $CONFIG --name $ARCH-scratch-fvg-0.9 --group $GROUP --batch_size $BATCH_SIZE --env $ENV --eval_config configs/evaluation/recognition.yaml --output_dir $OUTPUT_DIR
python evaluate.py --config_file $CONFIG --name $ARCH-scratch-fvg-1 --group $GROUP --batch_size $BATCH_SIZE --env $ENV --eval_config configs/evaluation/recognition.yaml --output_dir $OUTPUT_DIR

############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################

############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################

# FINE TUNE CASIA
python evaluate.py --config_file $CONFIG --name $ARCH-$DATASET-tuned-casia-1 --group $GROUP --batch_size $BATCH_SIZE --env $ENV --eval_config configs/evaluation/recognition.yaml --output_dir $OUTPUT_DIR
python evaluate.py --config_file $CONFIG --name $ARCH-$DATASET-tuned-casia-2 --group $GROUP --batch_size $BATCH_SIZE --env $ENV --eval_config configs/evaluation/recognition.yaml --output_dir $OUTPUT_DIR
python evaluate.py --config_file $CONFIG --name $ARCH-$DATASET-tuned-casia-3 --group $GROUP --batch_size $BATCH_SIZE --env $ENV --eval_config configs/evaluation/recognition.yaml --output_dir $OUTPUT_DIR
python evaluate.py --config_file $CONFIG --name $ARCH-$DATASET-tuned-casia-5 --group $GROUP --batch_size $BATCH_SIZE --env $ENV --eval_config configs/evaluation/recognition.yaml --output_dir $OUTPUT_DIR
python evaluate.py --config_file $CONFIG --name $ARCH-$DATASET-tuned-casia-7 --group $GROUP --batch_size $BATCH_SIZE --env $ENV --eval_config configs/evaluation/recognition.yaml --output_dir $OUTPUT_DIR
python evaluate.py --config_file $CONFIG --name $ARCH-$DATASET-tuned-casia-9 --group $GROUP --batch_size $BATCH_SIZE --env $ENV --eval_config configs/evaluation/recognition.yaml --output_dir $OUTPUT_DIR
python evaluate.py --config_file $CONFIG --name $ARCH-$DATASET-tuned-casia-10 --group $GROUP --batch_size $BATCH_SIZE --env $ENV --eval_config configs/evaluation/recognition.yaml --output_dir $OUTPUT_DIR

# # FINE TUNE FVG
python evaluate.py --config_file $CONFIG --name $ARCH-$DATASET-tuned-fvg-0.1 --group $GROUP --batch_size $BATCH_SIZE --env $ENV --eval_config configs/evaluation/recognition.yaml --output_dir $OUTPUT_DIR
python evaluate.py --config_file $CONFIG --name $ARCH-$DATASET-tuned-fvg-0.2 --group $GROUP --batch_size $BATCH_SIZE --env $ENV --eval_config configs/evaluation/recognition.yaml --output_dir $OUTPUT_DIR
python evaluate.py --config_file $CONFIG --name $ARCH-$DATASET-tuned-fvg-0.3 --group $GROUP --batch_size $BATCH_SIZE --env $ENV --eval_config configs/evaluation/recognition.yaml --output_dir $OUTPUT_DIR
python evaluate.py --config_file $CONFIG --name $ARCH-$DATASET-tuned-fvg-0.5 --group $GROUP --batch_size $BATCH_SIZE --env $ENV --eval_config configs/evaluation/recognition.yaml --output_dir $OUTPUT_DIR
python evaluate.py --config_file $CONFIG --name $ARCH-$DATASET-tuned-fvg-0.7 --group $GROUP --batch_size $BATCH_SIZE --env $ENV --eval_config configs/evaluation/recognition.yaml --output_dir $OUTPUT_DIR
python evaluate.py --config_file $CONFIG --name $ARCH-$DATASET-tuned-fvg-0.9 --group $GROUP --batch_size $BATCH_SIZE --env $ENV --eval_config configs/evaluation/recognition.yaml --output_dir $OUTPUT_DIR
python evaluate.py --config_file $CONFIG --name $ARCH-$DATASET-tuned-fvg-1 --group $GROUP --batch_size $BATCH_SIZE --env $ENV --eval_config configs/evaluation/recognition.yaml --output_dir $OUTPUT_DIR
