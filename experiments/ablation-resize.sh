#!/bin/bash

set -e
cd ..

# for debugging use `dryrun`, otherwise use `run`
WANDB_MODE=run

GROUP=resize-ablation
ENV=adrian-exodus

EPOCHS=200
BATCH_SIZE=256

TRAIN_TAIL="--dataset casia --group $GROUP --mode $WANDB_MODE --batch_size $BATCH_SIZE --epochs $EPOCHS --env $ENV"
EVAL_TAIL="--eval_config configs/evaluation/recognition.yaml --group $GROUP --batch_size $BATCH_SIZE --env $ENV --output_dir $GROUP"

#############################################################################################
#############################################################################################
#############################################################################################
python main.py --config_file configs/model_configs/simple_vit.yaml --name simplevit-linear $TRAIN_TAIL --model_args.resize linear
python main.py --config_file configs/model_configs/simple_vit.yaml --name simplevit-upsample-bicubic $TRAIN_TAIL --model_args.resize upsample --convert_tssi 1 --upsample_kind bicubic
python main.py --config_file configs/model_configs/simple_vit.yaml --name simplevit-upsample-bilinear $TRAIN_TAIL --model_args.resize upsample --convert_tssi 1 --upsample_kind bilinear
python main.py --config_file configs/model_configs/simple_vit.yaml --name simplevit-deconv $TRAIN_TAIL --model_args.resize deconv --convert_tssi 1

python main.py --config_file configs/model_configs/crossformer.yaml --name crossformer-linear $TRAIN_TAIL --model_args.resize linear
python main.py --config_file configs/model_configs/crossformer.yaml --name crossformer-upsample-bicubic $TRAIN_TAIL --model_args.resize upsample --convert_tssi 1 --upsample_kind bicubic
python main.py --config_file configs/model_configs/crossformer.yaml --name crossformer-upsample-bilinear $TRAIN_TAIL --model_args.resize upsample --convert_tssi 1 --upsample_kind bilinear
python main.py --config_file configs/model_configs/crossformer.yaml --name crossformer-deconv $TRAIN_TAIL --model_args.resize deconv --convert_tssi 1

python main.py --config_file configs/model_configs/t2t.yaml --name t2t-linear $TRAIN_TAIL --model_args.resize linear
python main.py --config_file configs/model_configs/t2t.yaml --name t2t-upsample-bicubic $TRAIN_TAIL --model_args.resize upsample --convert_tssi 1 --upsample_kind bicubic
python main.py --config_file configs/model_configs/t2t.yaml --name t2t-upsample-bilinear $TRAIN_TAIL --model_args.resize upsample --convert_tssi 1 --upsample_kind bilinear
python main.py --config_file configs/model_configs/t2t.yaml --name t2t-deconv $TRAIN_TAIL --model_args.resize deconv --convert_tssi 1

python main.py --config_file configs/model_configs/twins_svt.yaml --name twins-svt-linear $TRAIN_TAIL --model_args.resize linear
python main.py --config_file configs/model_configs/twins_svt.yaml --name twins-svt-upsample-bicubic $TRAIN_TAIL --model_args.resize upsample --convert_tssi 1 --upsample_kind bicubic
python main.py --config_file configs/model_configs/twins_svt.yaml --name twins-svt-upsample-bilinear $TRAIN_TAIL --model_args.resize upsample --convert_tssi 1 --upsample_kind bilinear
python main.py --config_file configs/model_configs/twins_svt.yaml --name twins-svt-deconv $TRAIN_TAIL --model_args.resize deconv --convert_tssi 1

python main.py --config_file configs/model_configs/cait.yaml --name cait-linear $TRAIN_TAIL --model_args.resize linear
python main.py --config_file configs/model_configs/cait.yaml --name cait-upsample-bicubic $TRAIN_TAIL --model_args.resize upsample --convert_tssi 1 --upsample_kind bicubic
python main.py --config_file configs/model_configs/cait.yaml --name cait-upsample-bilinear $TRAIN_TAIL --model_args.resize upsample --convert_tssi 1 --upsample_kind bilinear
python main.py --config_file configs/model_configs/cait.yaml --name cait-deconv $TRAIN_TAIL --model_args.resize deconv --convert_tssi 1

##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################

python evaluate.py --config_file configs/model_configs/simple_vit.yaml --name simplevit-linear $EVAL_TAIL --model_args.resize linear
python evaluate.py --config_file configs/model_configs/simple_vit.yaml --name simplevit-upsample-bicubic $EVAL_TAIL --model_args.resize upsample --convert_tssi 1 --upsample_kind bicubic
python evaluate.py --config_file configs/model_configs/simple_vit.yaml --name simplevit-upsample-bilinear $EVAL_TAIL --model_args.resize upsample --convert_tssi 1 --upsample_kind bilinear
python evaluate.py --config_file configs/model_configs/simple_vit.yaml --name simplevit-deconv $EVAL_TAIL --model_args.resize deconv --convert_tssi 1

python evaluate.py --config_file configs/model_configs/crossformer.yaml --name crossformer-linear $EVAL_TAIL --model_args.resize linear
python evaluate.py --config_file configs/model_configs/crossformer.yaml --name crossformer-upsample-bicubic $EVAL_TAIL --model_args.resize upsample --convert_tssi 1 --upsample_kind bicubic
python evaluate.py --config_file configs/model_configs/crossformer.yaml --name crossformer-upsample-bilinear $EVAL_TAIL --model_args.resize upsample --convert_tssi 1 --upsample_kind bilinear
python evaluate.py --config_file configs/model_configs/crossformer.yaml --name crossformer-deconv $EVAL_TAIL --model_args.resize deconv --convert_tssi 1

python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-linear $EVAL_TAIL --model_args.resize linear
python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-upsample-bicubic $EVAL_TAIL --model_args.resize upsample --convert_tssi 1 --upsample_kind bicubic
python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-upsample-bilinear $EVAL_TAIL --model_args.resize upsample --convert_tssi 1 --upsample_kind bilinear
python evaluate.py --config_file configs/model_configs/cait.yaml --name cait-deconv $EVAL_TAIL --model_args.resize deconv --convert_tssi 1

python evaluate.py --config_file configs/model_configs/twins_svt.yaml --name twins-svt-linear $EVAL_TAIL --model_args.resize linear
python evaluate.py --config_file configs/model_configs/twins_svt.yaml --name twins-svt-upsample-bicubic $EVAL_TAIL --model_args.resize upsample --convert_tssi 1 --upsample_kind bicubic
python evaluate.py --config_file configs/model_configs/twins_svt.yaml --name twins-svt-upsample-bilinear $EVAL_TAIL --model_args.resize upsample --convert_tssi 1 --upsample_kind bilinear
python evaluate.py --config_file configs/model_configs/twins_svt.yaml --name twins-svt-deconv $EVAL_TAIL --model_args.resize deconv --convert_tssi 1

python evaluate.py --config_file configs/model_configs/t2t.yaml --name t2t-linear $EVAL_TAIL --model_args.resize linear
python evaluate.py --config_file configs/model_configs/t2t.yaml --name t2t-upsample-bicubic $EVAL_TAIL --model_args.resize upsample --convert_tssi 1 --upsample_kind bicubic
python evaluate.py --config_file configs/model_configs/t2t.yaml --name t2t-upsample-bilinear $EVAL_TAIL --model_args.resize upsample --convert_tssi 1 --upsample_kind bilinear
python evaluate.py --config_file configs/model_configs/t2t.yaml --name t2t-deconv $EVAL_TAIL --model_args.resize deconv --convert_tssi 1
