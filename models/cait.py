from .hacked_models.cait import CaiT
from models.base_model import BaseModel
import constants

class GaitCaiT(BaseModel):
    def __init__(self, args):
        super(GaitCaiT, self).__init__(args)

        self.model = CaiT(
            image_size = (self.args.period_length, self.resize_to),
            patch_size = self.patch_size,
            channels = constants.NUM_CHANNELS,

            num_classes = self.args.model_args.embedding_size,

            dim = self.args.model_args.dim_size,
            heads = self.args.model_args.n_heads,
            mlp_dim = self.args.model_args.dim_size * 4,

            depth = self.args.model_args.num_layers,
            cls_depth = self.args.model_args.cls_depth,

            dropout = 0.0,
            emb_dropout = 0.0,
            layer_dropout = 0.0,
        )

