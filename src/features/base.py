import gc
import pandas as pd
import category_encoders as ce

from src.features.groupby import GroupbyIDTransformer
from src.features.date import CountTransaction, TransactionDays, P2Increase
from src.constant import CAT_FEATURES, DATE_FEATURES
from src.utils import reduce_mem_usage


def generate_features(features_df, label=None, encoder=None):
    cnt_features = [
        f for f in features_df.columns if f not in CAT_FEATURES + DATE_FEATURES + ['customer_ID']
    ]

    # Label Encoder
    if encoder is None:
        encoder = ce.OrdinalEncoder(cols=CAT_FEATURES, handle_unknown='impute')
        features_df = encoder.fit_transform(features_df)
    else:
        features_df = encoder.transform(features_df)

    for c in CAT_FEATURES:
        features_df[c] = features_df[c].astype('category')

    # Sort ID, S_2
    features_df = features_df.sort_values(by=['customer_ID', 'S_2'])

    transformers = [
        GroupbyIDTransformer(cnt_features, aggs=['max', 'min', 'mean', 'std', 'last']),
        GroupbyIDTransformer(CAT_FEATURES, aggs=['count', 'last']),
        TransactionDays(aggs=['max', 'min', 'mean', 'std']),
        P2Increase(aggs=['max', 'last']),
        CountTransaction(),
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

    df = reduce_mem_usage(df)

    return df, encoder
