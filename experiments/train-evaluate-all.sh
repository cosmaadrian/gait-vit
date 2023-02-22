#!/bin/bash
set -e

bash base_pretrain_finetune.sh simple_vit_bicubic simple_vit dense-gait
bash base_pretrain_finetune.sh simple_vit_bicubic simple_vit grew

bash base_pretrain_finetune.sh cait_bicubic cait grew
bash base_pretrain_finetune.sh cait_bicubic cait dense-gait

bash base_pretrain_finetune.sh t2t_bicubic t2t dense-gait
bash base_pretrain_finetune.sh t2t_bicubic t2t grew

bash base_pretrain_finetune.sh crossformer_bicubic crossformer dense-gait
bash base_pretrain_finetune.sh crossformer_bicubic crossformer grew

bash base_pretrain_finetune.sh twins_svt_bicubic twins_svt dense-gait
bash base_pretrain_finetune.sh twins_svt_bicubic twins_svt grew


###########################################################################33
###########################################################################33
###########################################################################33
###########################################################################33

bash base_scratch.sh simple_vit_bicubic simple_vit
bash base_scratch.sh cait_bicubic cait
bash base_scratch.sh t2t_bicubic t2t
bash base_scratch.sh crossformer_bicubic crossformer
bash base_scratch.sh twins_svt_bicubic twins_svt

bash base_evaluate.sh simple_vit_bicubic simple_vit dense-gait
bash base_evaluate.sh simple_vit_bicubic simple_vit grew

bash base_evaluate.sh cait_bicubic cait dense-gait
bash base_evaluate.sh cait_bicubic cait grew

bash base_evaluate.sh t2t_bicubic t2t dense-gait
bash base_evaluate.sh t2t_bicubic t2t grew

bash base_evaluate.sh crossformer_bicubic crossformer dense-gait
bash base_evaluate.sh crossformer_bicubic crossformer grew

bash base_evaluate.sh twins_svt_bicubic twins_svt dense-gait
bash base_evaluate.sh twins_svt_bicubic twins_svt grew
