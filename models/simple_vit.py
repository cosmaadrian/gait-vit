import constants
from vit_pytorch import SimpleViT
from models.base_model import BaseModel

class GaitViT(BaseModel):
    def __init__(self, args):
        super(GaitViT, self).__init__(args)

        self.model = SimpleViT(
            image_size = (self.args.period_length, self.resize_to),
            patch_size = self.patch_size,
            channels = constants.NUM_CHANNELS,

            num_classes = self.args.model_args.embedding_size,

            dim = self.args.model_args.dim_size,
            heads = self.args.model_args.n_heads,
            mlp_dim = self.args.model_args.dim_size * 4,

            depth = self.args.model_args.num_layers,
        )
