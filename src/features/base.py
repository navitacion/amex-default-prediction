import gc
import pandas as pd

from src.constant import CAT_FEATURES
from src.utils import reduce_mem_usage


def generate_features(features_df, transformers, label=None, logger=None, phase='train'):
    if logger is not None:
        logger.info('generate features')

    for c in CAT_FEATURES:
        features_df[c] = features_df[c].astype('category')

    # Sort ID, S_2
    features_df = features_df.sort_values(by=['customer_ID', 'S_2'])

    df = pd.DataFrame()

    for i, transformer in enumerate(transformers):

        logger.info(f'Execute Feature {transformer.__class__.__name__}')

        _feats = transformer(features_df, phase=phase)

        _feats = reduce_mem_usage(_feats)

        logger.info(f'Extracted Feature Shape {_feats.shape}')

        if i == 0:
            df = _feats.copy()
        else:
            df = pd.merge(df, _feats, on=['customer_ID'], how='left')

        del _feats
        gc.collect()

        logger.info(f'Data Shape {df.shape}')

    df = reduce_mem_usage(df)

    return df
