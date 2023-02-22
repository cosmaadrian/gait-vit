from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tqdm
import wandb
import os

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import nomenclature
import torch

from datasets import CASIADataset
from datasets.transforms import RandomCrop, ToTensor, DeterministicCrop

from lib.evaluator_extra import AcumenEvaluator

class CASIARecognitionEvaluator(AcumenEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dataloader = CASIADataset.val_dataloader(self.args)
        self.__train_embeddings = None

    @torch.no_grad()
    def trainer_evaluate(self, step):
        eval_results_A = self._evaluate_single(kind = 'A')
        eval_results_B = self._evaluate_single(kind = 'B')
        eval_results_C = self._evaluate_single(kind = 'C')

        val_acc = eval_results_A.mean()['Accuracy']
        val_acc_B = eval_results_B.mean()['Accuracy']
        val_acc_C = eval_results_C.mean()['Accuracy']

        log_dict = dict()
        for _, row in eval_results_A.iterrows():
            log_dict[f"Angle_{row['Probe Angle']}"] = row['Accuracy']

        self.logger.log_dict(log_dict, force_log = True, on_step = False)
        self._clear_cache()

        return {
            'NM': val_acc,
            'CB': val_acc_B,
            'CL': val_acc_C,
            'mean': (val_acc + val_acc_B + val_acc_C) / 3
        }

    @torch.no_grad()
    def evaluate(self, save = False):
        eval_results_A = self._evaluate_single(kind = 'A')
        eval_results_B = self._evaluate_single(kind = 'B')
        eval_results_C = self._evaluate_single(kind = 'C')

        eval_results_A['variation'] = 'NM'
        eval_results_B['variation'] = 'CL'
        eval_results_C['variation'] = 'CB'

        eval_results = pd.concat([eval_results_A, eval_results_B, eval_results_C])

        if save:
            eval_results.to_csv(f'results/{self.args.output_dir}/{self.args.group}:{self.args.name}-CASIA.csv', index = False)

        return eval_results

    def _clear_cache(self):
        self.__train_embeddings = None

    def _predict(self, annotations, kind = 'gallery'):
        dataloader = CASIADataset.val_dataloader(self.args, annotations = annotations)

        return torch.cat([
            self.model(data['image'].to(nomenclature.device))['representation'] for data in tqdm.tqdm(dataloader)
        ]).detach().cpu().numpy()

    def _evaluate_set(self, kind = 'A'):
        test_df = self.dataloader.dataset.annotations
        gallery_walks = test_df[test_df['type'] == 2]
        knn_train = gallery_walks.where(gallery_walks['run_id'] <= 3).dropna().sort_values(by = 'file_name')

        if kind == 'A':
            knn_test = gallery_walks.where(gallery_walks['run_id'] > 3).dropna().sort_values(by = 'file_name')
        elif kind == 'B':
            knn_test = test_df[test_df['type'] == 1].sort_values(by = 'file_name')
        elif kind == 'C':
            knn_test = test_df[test_df['type'] == 0].sort_values(by = 'file_name')
        else:
            raise Exception(f'{kind} is not a valid evaluation protocol.')

        knn_train = knn_train.reset_index(drop = True)
        knn_test = knn_test.reset_index(drop = True)

        with torch.no_grad():
            train_embeddings = self.__train_embeddings
            if train_embeddings is None:
                train_embeddings = self._predict(knn_train, kind = 'gallery')
                self.__train_embeddings = train_embeddings

            test_embeddings = self._predict(knn_test, kind = 'probe')

        results = {
            'Gallery Angle': [],
            'Probe Angle': [],
            'Accuracy': [],
        }

        for camera_id_gallery in sorted(knn_train['camera_id'].unique()):
            for camera_id_probe in sorted(knn_test['camera_id'].unique()):
                knn_gallery = knn_train[knn_train['camera_id'] == camera_id_gallery]
                knn_probe = knn_test[knn_test['camera_id'] == camera_id_probe]

                gallery_embeddings = train_embeddings[knn_gallery.index]
                probe_embeddings = test_embeddings[knn_probe.index]

                y_train = knn_gallery['track_id'].values
                y_test = knn_probe['track_id'].values

                knn = KNeighborsClassifier(1, p = 1)
                knn.fit(gallery_embeddings, y_train)

                y_pred = knn.predict(probe_embeddings)

                accuracy = (y_pred == y_test).mean()

                results['Gallery Angle'].append(camera_id_gallery)
                results['Probe Angle'].append(camera_id_probe)
                results['Accuracy'].append(accuracy)

        results = pd.DataFrame(results)
        return results

    def visualize(self):
        test_df = self.dataloader.dataset.annotations
        gallery_walks = test_df[test_df['type'] == 2]
        knn_train = gallery_walks.where(gallery_walks['run_id'] <= 3).dropna().sort_values(by = 'file_name')
        knn_train = knn_train.reset_index(drop = True)

        knn_test = gallery_walks.where(gallery_walks['run_id'] > 3).dropna().sort_values(by = 'file_name')
        knn_test = knn_test.reset_index(drop = True)

        with torch.no_grad():
            train_embeddings = self.__train_embeddings
            if train_embeddings is None:
                train_embeddings = self._predict(knn_train, kind = 'gallery')
                self.__train_embeddings = train_embeddings

            test_embeddings = self._predict(knn_test, kind = 'probe')

        # tsne = MDS(2, n_jobs = 15, verbose = 2, max_iter = 30)
        tsne = TSNE(2, verbose = 0, perplexity = 12, early_exaggeration = 32)
        embs = np.vstack((train_embeddings, test_embeddings))
        embs = StandardScaler().fit_transform(embs)
        _encoded = tsne.fit_transform(embs)
        encoded = _encoded[:len(train_embeddings)]
        encoded_test = _encoded[len(train_embeddings):]

        fig, ax = plt.subplots(1)
        # fig, ax = plt.subplots(2, 2)
        # ax[0, 0].set_title('CameraID')
        # ax[0, 0].scatter(x = encoded[:, 0], y = encoded[:, 1], c = knn_train['camera_id'].astype('category').cat.codes.values.ravel())
        # ax[1, 0].scatter(x = encoded_test[:, 0], y = encoded_test[:, 1], c = knn_test['camera_id'].astype('category').cat.codes.values.ravel(), marker = 'x')

        # ax[0, 1].set_title('TrackID')
        # ax[0, 1].scatter(x = encoded[:, 0], y = encoded[:, 1], c = knn_train['track_id'].astype('category').cat.codes.values.ravel())
        # ax[1, 1].scatter(x = encoded_test[:, 0], y = encoded_test[:, 1], c = knn_test['track_id'].astype('category').cat.codes.values.ravel(), marker ='x')
        ax.scatter(x = encoded[:, 0], y = encoded[:, 1], c = knn_train['track_id'].astype('category').cat.codes.values.ravel())

        return fig, ax

    def _evaluate_single(self,  kind = 'A', same_angle = False):
        test_df = self.dataloader.dataset.annotations
        gallery_walks = test_df[test_df['type'] == 2]
        knn_train = gallery_walks.where(gallery_walks['run_id'] <= 3).dropna().sort_values(by = 'file_name')

        if kind == 'A':
            knn_test = gallery_walks.where(gallery_walks['run_id'] > 3).dropna().sort_values(by = 'file_name')
        elif kind == 'B':
            knn_test = test_df[test_df['type'] == 1].sort_values(by = 'file_name')
        elif kind == 'C':
            knn_test = test_df[test_df['type'] == 0].sort_values(by = 'file_name')

        knn_train = knn_train.reset_index(drop = True)
        knn_test = knn_test.reset_index(drop = True)

        with torch.no_grad():
            train_embeddings = self.__train_embeddings
            if train_embeddings is None:
                train_embeddings = self._predict(knn_train, kind = 'gallery')
                self.__train_embeddings = train_embeddings

            test_embeddings = self._predict(knn_test, kind = 'probe')

        results = {
            'Probe Angle': [],
            'Accuracy': [],
        }

        knn_gallery = knn_train

        for camera_id_probe in sorted(knn_test['camera_id'].unique()):
            if not same_angle:
                knn_gallery = knn_train[knn_train['camera_id'] != camera_id_probe]

            gallery_embeddings = train_embeddings[knn_gallery.index]

            knn = KNeighborsClassifier(1, p = 1)
            knn.fit(gallery_embeddings, knn_gallery['track_id'].values)

            knn_probe = knn_test[knn_test['camera_id'] == camera_id_probe]

            probe_embeddings = test_embeddings[knn_probe.index]
            y_pred = knn.predict(probe_embeddings)

            accuracy = (y_pred == knn_probe['track_id'].values).mean()

            results['Probe Angle'].append(camera_id_probe)
            results['Accuracy'].append(accuracy)

        results = pd.DataFrame(results)
        return results
