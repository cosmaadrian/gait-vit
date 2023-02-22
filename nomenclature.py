import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

import models
MODELS = {
    'vit': models.GaitViT,
    'cait': models.GaitCaiT,
    't2t': models.GaitT2TViT,
    'twins-svt': models.GaitTwinsSVT,
    'crossformer': models.GaitCrossFormer
}

import datasets
DATASETS = {
    'dense-gait': datasets.DenseGait,
    'ouisir': datasets.OUISIRDataset,
    'grew': datasets.GREWDataset,

    'casia': datasets.CASIADataset,
    'fvg': datasets.FVGDataset,
}

import trainers
TRAINER = {
    'contrastive': trainers.ContrastiveTrainer,
    'fine-tuner': trainers.FineTunerTrainer,
}


import evaluators
EVALUATORS = {
    'casia-recognition': evaluators.CASIARecognitionEvaluator,
    'fvg-recognition': evaluators.FVGRecognitionEvaluator,
}
