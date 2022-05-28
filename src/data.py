import gc
from pathlib import Path
import pickle
import yaml
import pandas as pd
from tqdm import tqdm
from logging import getLogger, config

from .utils import reduce_mem_usage
from .features.tmp import tmp_features

class DataAsset:

    def __init__(self, cfg):
        self.cfg = cfg
        self.data_dir = Path(self.cfg.data.data_dir)

        with open('logging.yaml', 'r') as yml:
            logger_cfg = yaml.safe_load(yml)

        # 定義ファイルを使ったloggingの設定
        config.dictConfig(logger_cfg)

        # ロガーの取得
        self.logger = getLogger("Data Asset Logger")


    def _extract_train_data_from_specific_id(
            self, customer_ids: list, reduce_mem: bool = False):

        features = pd.DataFrame()
        feature_types = ['D', 'S', 'P', 'B', 'R']

        for i, s in tqdm(enumerate(feature_types), total=len(feature_types)):

            pickle_path = self.data_dir.joinpath(f'train_data_{s}.pkl')
            with open(pickle_path, 'rb') as f:
                _features = pickle.load(f)

            _features = _features[_features['customer_ID'].isin(customer_ids)]
            if reduce_mem:
                _features = reduce_mem_usage(_features)

            if i == 0:
                features = _features
            else:
                # PK: customer_ID + S_2
                features = pd.merge(features, _features, on=['customer_ID', 'S_2'])

            del _features
            gc.collect()

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
        features = self._extract_train_data_from_specific_id(unique_customer_ids)

        # TODO: このタイミングで特徴量計算をする
        # PK: customer_ID + 'S_2' -> PK: customer_IDにする
        features = tmp_features(features)

        df = pd.merge(features, label, on=['customer_ID'], how='left')

        self.logger.info(f'Loaded Train Data Shape: {df.shape}')

        # Check unique customer_ID num
        assert label.shape[0] == df['customer_ID'].nunique()

        del features, label
        gc.collect()

        return df


    def get_generator_loading_test(self, chunksize: int = 20000):

        test = pd.read_csv(self.data_dir.joinpath('test_data.csv'), chunksize=chunksize)

        return test





