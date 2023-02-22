import os
import pprint
import torch
import pandas as pd
import numpy as np

from datasets.skeleton_sequence import SkeletonSequenceDataset

class DenseGait(SkeletonSequenceDataset):

    def __init__(self, args, image_transforms = None, annotations = None, kind = None):
        self.args = args
        self.image_transforms = image_transforms

        self.annotation_path = os.path.join(args.environment['dense-gait'], 'annotations.csv')
        self.annotations = pd.read_csv(self.annotation_path)

        if 'fraction' in self.args and self.args.fraction is not None:
            ids = np.arange(len(self.annotations['track_id'].unique()))
            ids = np.random.choice(ids, size = int(self.args.fraction * len(ids)), replace = False)
            self.annotations = self.annotations[self.annotations['track_id'].isin(ids)]

        self.annotations.track_id = self.annotations.track_id.astype('category').cat.codes

    def __getitem__(self, idx):
        sample = self.annotations.iloc[idx]
        image = np.load(os.path.join(self.args.environment['dense-gait'], '../', sample['file_name']))

        assert not np.any(np.isnan(image))
        image[:, :, 1] = - image[:, :, 1]

        instantiated_sample = {
            'image': image,
            'track_id': np.array(sample['track_id']),
        }

        return self.apply_transforms(instantiated_sample)
