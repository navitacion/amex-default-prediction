from pathlib import Path
import pickle
import pandas as pd


class DataAsset:

    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.data_dir = Path(self.cfg.data.data_dir)
        self.logger = logger

    def _extract_train_data_from_specific_id(self, customer_ids: list):
        pickle_path = self.data_dir.joinpath('train_data_prep.pkl')
        with open(pickle_path, 'rb') as f:
            features = pickle.load(f)
        features = features[features['customer_ID'].isin(customer_ids)].reset_index(drop=True)

        return features

    def load_train_data(self):
        frac = self.cfg.data.sample_frac
        seed = self.cfg.data.seed

        self.logger.info('Start Train Data')

        label = pd.read_csv(self.data_dir.joinpath('train_labels.csv'))

        # Sampled
        if frac < 1.0:
            label = label.sample(frac=frac, random_state=seed)
        else:
            pass

        self.logger.info(f'Unique customer_ID Num: {label.shape[0]}')

        unique_customer_ids = label['customer_ID'].unique().tolist()

        # PK: customer_ID + 'S_2'
        train = self._extract_train_data_from_specific_id(unique_customer_ids)

        return train, label
