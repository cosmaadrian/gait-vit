import os
import torch
import pandas as pd
import numpy as np

from datasets.skeleton_sequence import SkeletonSequenceDataset

class GREWDataset(SkeletonSequenceDataset):
    def __init__(self, args, kind = 'train', annotations = None, image_transforms = None):
        self.args = args
        self.kind = kind
        self.image_transforms = image_transforms

        self.annotation_path = os.path.join(self.args.environment['grew'], f'annotations_{kind}.csv')
        self.annotations = pd.read_csv(self.annotation_path)

        self.annotations.track_id = self.annotations.track_id.astype('category').cat.codes

    def __getitem__(self, idx):
        sample = self.annotations.iloc[idx]
        image = np.load(os.path.join(self.args.environment['grew'], self.kind, 'poses', sample['file_name']))

        assert not np.any(np.isnan(image))
        image[:, :, 1] = -image[:, :, 1]

        instantiated_sample = {
            'image': image,
            'track_id': np.array(sample['track_id']),
        }

        return self.apply_transforms(instantiated_sample)
