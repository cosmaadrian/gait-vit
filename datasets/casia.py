import os
import json
import torch
import pandas as pd
import numpy as np

from datasets.skeleton_sequence import SkeletonSequenceDataset
import constants
import utils


def get_casia(args, kind, train_split = 62):
    if train_split is None:
        train_split = 62

    df = pd.read_csv(os.path.join(args.environment['casia'], 'annotations.csv'))
    df['variation'] = df['type'].astype(str) + '-' + df['camera_id'].astype(str)
    df['variation'] = df['variation'].apply(lambda x: utils.casia_variation2idx[x])

    # df = df[df['camera_id'] == 54]
    df['track_id'] = df['track_id'].astype('category').cat.codes
    df['type'] = df['type'].astype('category').cat.codes
    df['run_id'] = df['run_id'].astype('category').cat.codes

    train_df = df[df['track_id'].isin(np.arange(0, train_split))]
    val_df = df[df['track_id'].isin(np.arange(train_split, 124))]

    if kind == 'train':
        return train_df

    elif kind in ['test', 'val']:
        return val_df

    return train_df, val_df

class CASIADataset(SkeletonSequenceDataset):

    def __init__(self, args, annotations = None, kind = 'train', image_transforms = None):
        self.args = args

        if annotations is None:
            self.annotations = get_casia(args = args, kind = kind)
        else:
            self.annotations = annotations

        if kind == 'train' and ('fraction' in self.args and self.args.fraction is not None):
            self.annotations = self.annotations.groupby('track_id').apply(lambda x: x.sample(frac = args.fraction))
            self.annotations = self.annotations.reset_index(drop = True)

        if kind == 'train' and ('runs' in self.args and self.args.runs is not None):
            self.annotations = self.annotations.groupby(['camera_id', 'track_id']).apply(lambda x: x.sample(n = self.args.runs, replace = False))
            self.annotations = self.annotations.reset_index(drop = True)

        self.image_transforms = image_transforms

    def __getitem__(self, idx):
        sample = self.annotations.iloc[idx]
        image = np.load(os.path.join(self.args.environment['casia'], sample['file_name']))

        instantiated_sample = {
            'image': image,
            'track_id': sample['track_id'].reshape((1, )),
            'camera_id': sample['camera_id'].reshape((1, )),
            'type': sample['type'].reshape((1, )),
            'run_id': sample['run_id'].reshape((1, )),
        }

        return self.apply_transforms(instantiated_sample)
