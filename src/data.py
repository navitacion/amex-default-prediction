import gc
from pathlib import Path
import pickle
import pandas as pd

from src.constant import DROP_FEATURES


# Load Dataset  ------------------------------------------------------------------------
class DataAsset:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.data_dir = Path(self.cfg.data.data_dir)
        self.logger = logger

    def _extract_data_from_specific_id(self, customer_ids: list, _type: str = 'train'):

        features = pd.DataFrame()
        feature_types = ['D', 'S', 'P', 'B', 'R']

        for i, s in enumerate(feature_types):
            pickle_path = self.data_dir.joinpath(f'{_type}_data_prep_{s}.pkl')
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

    def _get_test_customer_id(self):

        with open(self.data_dir.joinpath('test_customer_ids.pkl'), 'rb') as f:
            test_customer_ids = pickle.load(f)

        test_customer_ids = test_customer_ids['customer_ID'].values.tolist()

        return test_customer_ids

    def _split_array(self, _data: list, n_group: int):
        for i_chunk in range(n_group):
            yield _data[i_chunk * len(_data) // n_group:(i_chunk + 1) * len(_data) // n_group]

    def train_generator(self):
        # train dataの読み込み
        self.label = pd.read_csv(self.data_dir.joinpath('train_labels.csv'))
        # Sampled
        if self.cfg.data.sample_frac < 1.0:
            self.label = self.label.sample(frac=self.cfg.data.sample_frac, random_state=self.cfg.data.seed)
        else:
            pass

        # chunksizeごとに出力
        train_ids = self.label['customer_ID'].unique().tolist()
        chunksize = self.cfg.train.chunk_size

        for target_ids in self._split_array(train_ids, n_group=chunksize):
            train = self._extract_data_from_specific_id(target_ids, _type='train')
            train = train.drop(DROP_FEATURES, axis=1)

            yield train

    def test_generator(self):
        # test dataの読み込み
        test_ids = self._get_test_customer_id()
        chunksize = self.cfg.inference.chunk_size

        for target_ids in self._split_array(test_ids, n_group=chunksize):
            test = self._extract_data_from_specific_id(target_ids, _type='test')
            test = test.drop(DROP_FEATURES, axis=1)

            yield test, target_ids
