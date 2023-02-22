from lib.trainer_extra import AcumenTrainer
from trainers.losses import SupConLoss

class ContrastiveTrainer(AcumenTrainer):

    def __init__(self, args, model):
        super().__init__(args, model)
        self.criterion = SupConLoss(temperature = self.args.loss_args.temperature)

    def training_step(self, batch, batch_idx):
        output = self.model(batch['image'])

        features = output['projection'].view(-1, self.args.num_views, output['projection'].shape[-1])
        labels = batch['track_id'].squeeze()[::self.args.num_views]

        contrastive_loss = self.criterion(features, labels.squeeze())
        self.log('train/loss:recognition', contrastive_loss.item(), on_step = True)
        return contrastive_loss
