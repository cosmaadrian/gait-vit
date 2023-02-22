from vit_pytorch.twins_svt import TwinsSVT
from models.base_model import BaseModel

class GaitTwinsSVT(BaseModel):
    def __init__(self, args):
        super(GaitTwinsSVT, self).__init__(args)
        self.model = TwinsSVT(
            num_classes = self.args.model_args.embedding_size,       # number of output classes
            s1_emb_dim = self.args.model_args.dim_size[0],          # stage 1 - patch embedding projected dimension
            s1_patch_size = self.args.model_args.patch_size[0],        # stage 1 - patch size for patch embedding
            s1_local_patch_size = self.args.model_args.local_patch_size[0],  # stage 1 - patch size for local attention
            s1_global_k = self.args.model_args.global_k[0],          # stage 1 - global attention key / value reduction factor, defaults to 7 as specified in paper
            s1_depth = self.args.model_args.num_layers[0],             # stage 1 - number of transformer blocks (local attn -> ff -> global attn -> ff)

            s2_emb_dim = self.args.model_args.dim_size[1],         # stage 2 (same as above)
            s2_patch_size = self.args.model_args.patch_size[1],
            s2_local_patch_size = self.args.model_args.local_patch_size[1],
            s2_global_k = self.args.model_args.global_k[1],
            s2_depth = self.args.model_args.num_layers[1],

            s3_emb_dim = self.args.model_args.dim_size[2],         # stage 3 (same as above)
            s3_patch_size = self.args.model_args.patch_size[2],
            s3_local_patch_size = self.args.model_args.local_patch_size[2],
            s3_global_k = self.args.model_args.global_k[2],
            s3_depth = self.args.model_args.num_layers[2],

            s4_emb_dim = self.args.model_args.dim_size[3],         # stage 4 (same as above)
            s4_patch_size = self.args.model_args.patch_size[3],
            s4_local_patch_size = self.args.model_args.local_patch_size[3],
            s4_global_k = self.args.model_args.global_k[3],
            s4_depth = self.args.model_args.num_layers[3],

            peg_kernel_size = 3,      # positional encoding generator kernel size
            dropout = 0.1              # dropout
        )
