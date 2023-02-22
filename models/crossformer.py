from vit_pytorch.crossformer import CrossFormer
from models.base_model import BaseModel

class GaitCrossFormer(BaseModel):
    def __init__(self, args):
        super(GaitCrossFormer, self).__init__(args)
        self.model = CrossFormer(
            num_classes = self.args.model_args.embedding_size,
            dim = tuple(self.args.model_args.dim_size),         # dimension at each stage
            depth = tuple(self.args.model_args.num_layers),              # depth of transformer at each stage
            global_window_size = tuple(self.args.model_args.global_window_size), # global window sizes at each stage
            local_window_size = self.args.model_args.local_window_size,             # local window size (can be customized for each stage, but in paper, held constant at 7 for all stages)
            cross_embed_strides = tuple(self.args.model_args.cross_embed_strides),
            cross_embed_kernel_sizes = tuple(self.args.model_args.cross_embed_kernel_sizes),
        )
