import constants
from .hacked_models.t2t import T2TViT

from models.base_model import BaseModel


class GaitT2TViT(BaseModel):
    def __init__(self, args):
        super(GaitT2TViT, self).__init__(args)

        self.model = T2TViT(
            image_size = (self.args.period_length, self.resize_to),
            channels = constants.NUM_CHANNELS,

            num_classes = self.args.model_args.embedding_size,

            dim = self.args.model_args.dim_size,
            heads = self.args.model_args.n_heads,
            mlp_dim = self.args.model_args.dim_size * 4,

            depth = self.args.model_args.num_layers,

            t2t_layers = self.args.model_args.t2t_layers,
        )


