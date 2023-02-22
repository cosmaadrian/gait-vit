import os
import json
import torch
import pandas as pd
import numpy as np

from datasets.skeleton_sequence import SkeletonSequenceDataset
import constants
import utils


def get_fvg(args, kind, train_split = 136):
    if train_split is None:
        train_split = 136

    df = pd.read_csv(os.path.join(args.environment['fvg'], 'annotations.csv'))

    df['track_id'] = df['track_id'].astype('category').cat.codes
    df['run_id'] = df['run_id'].astype('category').cat.codes
    df['session_id'] = df['session_id'].astype('category').cat.codes
    df['variation'] = None

    df.loc[df[(df['run_id'].isin([0, 1, 2]))].index, 'variation'] = 'nm'

    df.loc[pd.concat([
        (df[(df['session_id'] == 0) & (df['run_id'].isin([3, 4, 5, 6, 7, 8]))]),
        (df[(df['session_id'] == 1) & (df['run_id'].isin([3, 4, 5]))]),
        (df[(df['session_id'] == 2) & (df['run_id'].isin([3, 4, 5]))]),
    ]).index, 'variation'] = 'ws'

    df.loc[pd.concat([
        (df[(df['session_id'] == 0) & (df['run_id'].isin([9, 10, 11]))])
    ]).index, 'variation'] = 'cb'

    df.loc[pd.concat([
        (df[(df['session_id'] == 1) & (df['run_id'].isin([6, 7, 8]))]),
        (df[(df['session_id'] == 2) & (df['run_id'].isin([6, 7, 8]))]),
    ]).index, 'variation'] = 'cl'

    df.loc[pd.concat([
        (df[(df['session_id'] == 1) & (df['run_id'].isin([9, 10, 11]))]),
        (df[ (df['session_id'] == 2) & (df['run_id'].isin([9, 10, 11]))]),
    ]).index, 'variation'] = 'cbg'

    df['variation'] = df['variation'].apply(lambda x: utils.fvg_variation2idx[x])

    train_df = df[df['track_id'].isin(np.arange(0, train_split))]
    val_df = df[df['track_id'].isin(np.arange(train_split, 225))]

    if kind == 'train':
        return train_df

    elif kind in ['test', 'val']:
        return val_df

    return train_df, val_df

class FVGDataset(SkeletonSequenceDataset):
    def __init__(self, args, annotations = None, kind = 'train', image_transforms = None):
        self.args = args
        self.annotations = annotations
        self.image_transforms = image_transforms

        if self.annotations is None:
            self.annotations = get_fvg(args = args, kind = kind)

        if kind == 'train' and 'fraction' in self.args and self.args.fraction is not None:
            self.annotations = self.annotations.groupby('track_id').apply(lambda x: x.sample(frac = args.fraction))
            self.annotations = self.annotations.reset_index(drop = True)


    def __getitem__(self, idx):
        sample = self.annotations.iloc[idx]
        path = os.path.join(self.args.environment['fvg'], sample['file_name'])
        image = np.load(path)

        instantiated_sample = {
            'image': image,
            'track_id': sample['track_id'].reshape((1, )),
            'session_id': sample['session_id'].reshape((1, )),
            'run_id': sample['run_id'].reshape((1, )),
            'variation': sample['variation'].reshape((1, )),
        }

        return self.apply_transforms(instantiated_sample)
