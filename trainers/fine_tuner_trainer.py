import torch
from lib.trainer_extra import AcumenTrainer
from trainers.losses import SupConLoss
from .contrastive_trainer import ContrastiveTrainer

class FineTunerTrainer(ContrastiveTrainer):

    def __init__(self, args, model):
        super().__init__(args, model)
        self.criterion = SupConLoss(temperature = self.args.loss_args.temperature)

    # def configure_optimizers(self, lr = 1e-6):
    #     if self._optimizer is not None:
    #         return self._optimizer

    #     #### LLRD
    #     print(":::: SA MOARA JAMILLA")

    #     opt_parameters = []
    #     named_parameters = list(self.model.named_parameters())

    #     no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    #     init_lr = 0.00001
    #     head_lr = 0.0001
    #     lr = init_lr

    #     # === Model Head ======================================================
    #     params_0 = [p for n,p in named_parameters if ("out_appearance" in n or "projection" in n or 'out_embedding' in n) and any(nd in n for nd in no_decay)]
    #     params_1 = [p for n,p in named_parameters if ("out_appearance" in n or "projection" in n or 'out_embedding' in n) and not any(nd in n for nd in no_decay)]

    #     opt_parameters.append({"params": params_0, "lr": head_lr, "weight_decay": 0.0})
    #     opt_parameters.append({"params": params_1, "lr": head_lr, "weight_decay": 0.01})

    #     for layer in range(self.args.model_args['num_layers'], -1, -1):
    #         params_0 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n and any(nd in n for nd in no_decay)]
    #         params_1 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n and not any(nd in n for nd in no_decay)]
    #         opt_parameters.append({"params": params_0, "lr": lr, "weight_decay": 0.0})
    #         opt_parameters.append({"params": params_1, "lr": lr, "weight_decay": 0.01})
    #         lr *= 0.9

    #     # === Embeddings layer ==========================================================
    #     params_0 = [p for n,p in named_parameters if "skeleton_encoding" in n and any(nd in n for nd in no_decay)]
    #     params_1 = [p for n,p in named_parameters if "skeleton_encoding" in n and not any(nd in n for nd in no_decay)]
    #     opt_parameters.append({"params": params_0, "lr": lr, "weight_decay": 0.0})
    #     opt_parameters.append({"params": params_1, "lr": lr, "weight_decay": 0.01})

    #     params_0 = [p for n,p in named_parameters if "positional_embedding" in n and any(nd in n for nd in no_decay)]
    #     params_1 = [p for n,p in named_parameters if "positional_embedding" in n and not any(nd in n for nd in no_decay)]
    #     opt_parameters.append({"params": params_0, "lr": lr, "weight_decay": 0.0})
    #     opt_parameters.append({"params": params_1, "lr": lr, "weight_decay": 0.01})

    #     self._optimizer = torch.optim.AdamW(opt_parameters, lr=init_lr)
    #     return self._optimizer
