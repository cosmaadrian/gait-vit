from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tqdm
import pprint
import nomenclature
import wandb

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch

from datasets import FVGDataset
from datasets.transforms import RandomCrop, ToTensor, DeterministicCrop

from lib.evaluator_extra import AcumenEvaluator


class FVGRecognitionEvaluator(AcumenEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataloader = FVGDataset.val_dataloader(self.args)

    @torch.no_grad()
    def trainer_evaluate(self, step):
        return {
            'ALL': self.evaluate_all()
        }

    @torch.no_grad()
    def evaluate(self, save = False):
        log_dict = {
            'WS': self.evaluate_walking_speed(),
            'CB': self.evaluate_carrying_bag(),
            'CL': self.evaluate_changing_clothes(),
            'CBG': self.evaluate_cluttered_background(),
            'ALL': self.evaluate_all(),
            # 'NM': self.evaluate_nm(), # dosen't work??
        }

        df = pd.DataFrame(log_dict, index = [0])

        if save:
            df.to_csv(f'results/{self.args.output_dir}/{self.args.group}_{self.args.name}-FVG.csv', index = False)

        return df

    def _predict(self, annotations):
        dataloader = FVGDataset.val_dataloader(self.args, annotations = annotations)

        return torch.cat([
            self.model(data['image'].to(nomenclature.device))['representation'] for data in tqdm.tqdm(dataloader)
        ]).detach().cpu().numpy()

    def evaluate_protocol(self, gallery_walks, probe_walks):
        gallery_walks = gallery_walks.reset_index(drop = True)
        probe_walks = probe_walks.reset_index(drop = True)

        with torch.no_grad():
            gallery_embeddings = self._predict(gallery_walks)
            probe_embeddings = self._predict(probe_walks)

        knn = KNeighborsClassifier(1, p = 1)
        knn.fit(gallery_embeddings, gallery_walks['track_id'].values)

        return knn.score(probe_embeddings, probe_walks['track_id'].values)

    def evaluate_carrying_bag(self):
        test_df = self.dataloader.dataset.annotations
        gallery_walks = test_df[
            (test_df['session_id'] == 0) &
            (test_df['run_id'] == 1)
        ]

        probe_walks = test_df[
            (test_df['session_id'] == 0) &
            (test_df['run_id'].isin([9, 10, 11]))
        ]

        return self.evaluate_protocol(gallery_walks, probe_walks)

    def evaluate_changing_clothes(self):
        test_df = self.dataloader.dataset.annotations
        gallery_walks = test_df[
            (test_df['session_id'] == 1) &
            (test_df['run_id'] == 1)
        ]

        probe_walks = pd.concat([
            (test_df[
                (test_df['session_id'] == 1) &
                (test_df['run_id'].isin([6, 7, 8]))]
            ),
            (test_df[
                (test_df['session_id'] == 2) &
                (test_df['run_id'].isin([6, 7, 8]))]
            )
        ])

        return self.evaluate_protocol(gallery_walks, probe_walks)

    def evaluate_cluttered_background(self):
        test_df = self.dataloader.dataset.annotations
        gallery_walks = test_df[
            (test_df['session_id'] == 1) &
            (test_df['run_id'] == 1)
        ]

        probe_walks = pd.concat([
            (test_df[
                (test_df['session_id'] == 1) &
                (test_df['run_id'].isin([9, 10, 11]))]
            ), (test_df[
                (test_df['session_id'] == 2) &
                (test_df['run_id'].isin([9, 10, 11]))]
            ),
        ])

        return self.evaluate_protocol(gallery_walks, probe_walks)

    def evaluate_nm(self):
        test_df = self.dataloader.dataset.annotations
        gallery_walks = pd.concat([
            (test_df[
                (test_df['session_id'] == 0) &
                (test_df['run_id'] == 1)]
            )
        ])

        probe_walks = pd.concat([
            (test_df[
                (test_df['session_id'] == 0) &
                (test_df['run_id'].isin([0, 2]))]
            ), (test_df[
                (test_df['session_id'] == 1) &
                (test_df['run_id'].isin([1, 2, 3]))]
            ),
             (test_df[
                (test_df['session_id'] == 2) &
                (test_df['run_id'].isin([1, 2, 3]))])
        ])

        return self.evaluate_protocol(gallery_walks, probe_walks)

    def evaluate_all(self):
        test_df = self.dataloader.dataset.annotations
        gallery_walks = pd.concat([
            (test_df[
                (test_df['session_id'] == 0) &
                (test_df['run_id'] == 1)]
            ),(test_df[
                (test_df['session_id'] == 1) &
                (test_df['run_id'] == 1)]
            )
        ])

        probe_walks = pd.concat([
            (test_df[
                (test_df['session_id'] == 0) &
                (test_df['run_id'].isin([3, 4, 5, 6, 7, 8]))]
            ), (test_df[
                (test_df['session_id'] == 1) &
                (test_df['run_id'].isin([3, 4, 5]))]
            ), (test_df[
                (test_df['session_id'] == 2) &
                (test_df['run_id'].isin([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))]
            )
        ])
        return self.evaluate_protocol(gallery_walks, probe_walks)

    def evaluate_walking_speed(self):
        test_df = self.dataloader.dataset.annotations
        gallery_walks = pd.concat([
            (test_df[
                (test_df['session_id'] == 0) &
                (test_df['run_id'] == 1)]
            ),(test_df[
                (test_df['session_id'] == 1) &
                (test_df['run_id'] == 1)]
            )
        ])

        probe_walks = pd.concat([
            (test_df[
                (test_df['session_id'] == 0) &
                (test_df['run_id'].isin([3, 4, 5, 6, 7, 8]))]
            ), (test_df[
                (test_df['session_id'] == 1) &
                (test_df['run_id'].isin([3, 4, 5]))]
            ),
             (test_df[
                (test_df['session_id'] == 2) &
                (test_df['run_id'].isin([3, 4, 5]))])
        ])
        return self.evaluate_protocol(gallery_walks, probe_walks)
