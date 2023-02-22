import torch
import torch.nn as nn
import constants
from einops.layers.torch import Rearrange

class LearnableResize(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        num_joints = constants.NUM_JOINTS if not args.convert_tssi else 39
        print("::: NUM_JOINTS:", num_joints)

        if args.model_args.resize == 'linear':
            self.resize = nn.Linear(num_joints, args.model_args.resize_to)

        elif args.model_args.resize == 'upsample':
            self.resize = nn.Upsample(size=(args.period_length, args.period_length), mode=args.upsample_kind, align_corners = True)

        elif args.model_args.resize == 'deconv':
           self.resize= nn.Sequential(
                Rearrange('b c h w -> b (c h) w'),
                nn.ConvTranspose1d(
                    in_channels = args.period_length * constants.NUM_CHANNELS,
                    out_channels = args.period_length * constants.NUM_CHANNELS,
                    kernel_size = 4,
                    stride = 2,
                ),
                nn.BatchNorm1d(constants.NUM_CHANNELS * args.period_length),
                nn.GELU(),
                nn.Conv1d(
                    in_channels = args.period_length * constants.NUM_CHANNELS,
                    out_channels = args.period_length * constants.NUM_CHANNELS,
                    kernel_size = 7,
                ),
                nn.BatchNorm1d(constants.NUM_CHANNELS * args.period_length),
                nn.GELU(),
                nn.Conv1d(
                    in_channels = args.period_length * constants.NUM_CHANNELS,
                    out_channels = args.period_length * constants.NUM_CHANNELS,
                    kernel_size = 7,
                ),
                nn.BatchNorm1d(constants.NUM_CHANNELS * args.period_length),
                nn.GELU(),
                nn.Conv1d(
                    in_channels = args.period_length * constants.NUM_CHANNELS,
                    out_channels = args.period_length * constants.NUM_CHANNELS,
                    kernel_size = 5,
                ),
                Rearrange('b (c h) w -> b c h w', c = constants.NUM_CHANNELS)
            )
        else:
            raise Exception(f"Unknown resize type: {args.model_args.resize}")

    def forward(self, sequence):
        resized = self.resize(sequence)
        return resized
