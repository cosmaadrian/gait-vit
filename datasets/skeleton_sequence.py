from torch.utils.data import Dataset, DataLoader

from datasets.transforms import RandomCrop, ToTensor, SqueezeAndFlip, Permutation, RandomPace, DeterministicCrop
from datasets.transforms import FlipSequence, DropOutFrames, DropOutJoints, JointNoise, PointNoise, ConvertToTSSI, CustomResize

from datasets.samplers import ContrastiveSampler, TwoViewsSampler
from torchvision import transforms
import constants

from lib.dataset_extra import AcumenDataset

class SkeletonSequenceDataset(AcumenDataset):
    def __len__(self):
        return len(self.annotations.index)

    def apply_transforms(self, instantiated_sample):
        if self.image_transforms:
            instantiated_sample = self.image_transforms(instantiated_sample)

        assert instantiated_sample['image'] is not None
        instantiated_sample['image'] = instantiated_sample['image'][:, :, :constants.NUM_CHANNELS]
        return instantiated_sample

    @property
    def num_classes(self):
        return len(self.annotations.track_id.unique())

    @classmethod
    def train_dataloader(cls, args, annotations = None):
        composed = transforms.Compose([
            RandomCrop(period_length = max(args.augmentation_args.paces) * args.period_length),
            RandomPace(paces = args.augmentation_args.paces, period_length = args.period_length),
            SqueezeAndFlip(amount = args.augmentation_args.squeeze_and_flip_amount, flip_prob = args.augmentation_args.flip_prob),
            FlipSequence(probability = args.augmentation_args.flip_sequence_prob),
            JointNoise(std = args.augmentation_args.joint_noise_std),
            PointNoise(std = args.augmentation_args.point_noise_std),
            DropOutJoints(prob = args.augmentation_args.drop_out_joints_prob, dropout_rate_range = args.augmentation_args.drop_out_joints_rate_range),
            ConvertToTSSI(do_apply = bool(args.convert_tssi)),
            ToTensor()
        ])
        dataset = cls(args = args, image_transforms = composed, annotations = annotations, kind = 'train')

        if args.sampler == 'two-views':
            sampler = TwoViewsSampler(args, dataset)
        else:
            sampler = None

        return DataLoader(
            dataset,
            batch_sampler = sampler,
            num_workers = args.environment.extra_args.num_workers,
            pin_memory = True,
            batch_size = (args.batch_size if sampler is None else 1)
        )

    @classmethod
    def val_dataloader(cls, args, annotations = None):
        composed = transforms.Compose([
            DeterministicCrop(period_length = args.period_length),
            ConvertToTSSI(do_apply = bool(args.convert_tssi)),
            ToTensor()
        ])

        dataset = cls(args = args, image_transforms = composed, annotations = annotations, kind = 'val')

        return DataLoader(
            dataset,
            batch_size = args.batch_size,
            shuffle = False,
            num_workers = args.environment.extra_args.num_workers,
            pin_memory = True,
        )

    def on_epoch_end(self):
        self.annotations = self.annotations.sample(frac = 1.0).reset_index(drop = True)
