import torch
import torch.nn as nn
import torch.nn.functional as F
import constants
from vit_pytorch import SimpleViT
from einops.layers.torch import Rearrange
from models.modules import LearnableResize

class BaseModel(torch.nn.Module):

    def __init__(self, args):
        super(BaseModel, self).__init__()

        self.args = args
        self.INPUT_SHAPE = (1, self.args.period_length, constants.NUM_JOINTS, constants.NUM_CHANNELS)

        if 'patch_h' in self.args.model_args and 'patch_w' in self.args.model_args:
            self.patch_size = (self.args.model_args.patch_h, self.args.model_args.patch_w)

        if self.args.model_args.resize != 'linear':
            self.resize_to = self.args.period_length
        else:
            self.resize_to = self.args.model_args.resize_to

        self.model = None # should be defined in the child class

        self.initial_processing = nn.Sequential(
            Rearrange('b h w c -> b c h w'),
            nn.BatchNorm2d(constants.NUM_CHANNELS),
        )

        self.resize = LearnableResize(args)

        self.projection = nn.Linear(
            in_features = self.args.model_args.embedding_size,
            out_features = self.args.model_args.projection_size,
        )


    def forward(self, sequence):
        sequence = self.initial_processing(sequence)
        sequence = self.resize(sequence)
        embedding = self.model(sequence)

        normalized_embedding = F.normalize(embedding)
        projection = F.normalize(self.projection(F.gelu(embedding)))

        output = {
            'representation': normalized_embedding,
            'projection': projection
        }


        return output
