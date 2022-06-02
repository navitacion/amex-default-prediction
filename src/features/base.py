import gc
import pandas as pd

from src.features.groupby import GroupbyIDTransformer
from src.features.date import RecentDateRecord
from src.constant import CAT_FEATURES, DATE_FEATURES


def generate_features(features_df, label=None):
    cnt_features = [
        f for f in features_df.columns if f not in CAT_FEATURES + DATE_FEATURES + ['customer_ID']
    ]

    transformers = [
        GroupbyIDTransformer(cnt_features, aggs=['max', 'min', 'mean']),
        RecentDateRecord()
    ]

    df = pd.DataFrame()

    for i, transformer in enumerate(transformers):

        _feats = transformer(features_df)

        if i == 0:
            if label is not None:
                df = pd.merge(label, _feats, on=['customer_ID'], how='left')
            else:
                df = _feats.copy()
        else:
            df = pd.merge(df, _feats, on=['customer_ID'], how='left')

        del _feats
        gc.collect()

    return df
