import os
import gc
import wandb
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from src.features.base import generate_features


class InferenceScoring:
    def __init__(self, cfg, models: list, logger, transformers):
        self.cfg = cfg
        self.models = models
        self.data_dir = Path(self.cfg.data.data_dir)
        self.logger = logger
        self.transformers = transformers

    def _get_generator_test_customer_id(self):

        with open(self.data_dir.joinpath('test_customer_ids.pkl'), 'rb') as f:
            test_customer_ids = pickle.load(f)

        test_customer_ids = test_customer_ids['customer_ID'].values.tolist()

        return test_customer_ids

    def _split_array(self, _data: list, n_group: int):
        for i_chunk in range(n_group):
            yield _data[i_chunk * len(_data) // n_group:(i_chunk + 1) * len(_data) // n_group]

    def _extract_train_data_from_specific_id(self, customer_ids: list):

        features = pd.DataFrame()
        feature_types = ['D', 'S', 'P', 'B', 'R']

        for i, s in enumerate(feature_types):

            pickle_path = self.data_dir.joinpath(f'test_data_prep_{s}.pkl')
            with open(pickle_path, 'rb') as f:
                _features = pickle.load(f)

            _features = _features[_features['customer_ID'].isin(customer_ids)]

            if i == 0:
                features = _features
            else:
                # PK: customer_ID + S_2
                features = pd.merge(features, _features, on=['customer_ID', 'S_2'])

            del _features
            gc.collect()

        return features

    def run(self):
        self.logger.info('Start Inference')

        ids = []
        preds = []

        all_customer_id_test = self._get_generator_test_customer_id()

        # Split Test Data Each customer_ID
        for target_ids in tqdm(
                self._split_array(all_customer_id_test, n_group=self.cfg.inference.chunk_size),
                total=self.cfg.inference.chunk_size):

            org_features_df = self._extract_train_data_from_specific_id(target_ids)

            # Feature Extract  -----------------------------------------
            df = generate_features(org_features_df, self.transformers, logger=self.logger, phase='predict')
            # Memory Clear
            del org_features_df
            gc.collect()

            # Inference  -----------------------------------------
            pred = np.zeros(len(df))
            for model in self.models:
                pred += model.predict(df[model.features])

            # Avg
            pred /= len(self.models)

            ids.extend(target_ids)
            preds.extend(pred)

            # Memory Clear
            del df, target_ids, pred
            gc.collect()

        res = pd.DataFrame({
            'customer_ID': ids,
            'prediction': preds
        })

        # Save to wandb
        sub_name = 'submission.csv'
        res.to_csv(os.path.join(self.cfg.data.asset_dir, sub_name), index=False)
        wandb.save(os.path.join(self.cfg.data.asset_dir, sub_name))

        return None
