import gc
import pandas as pd
import category_encoders as ce

from src.constant import CAT_FEATURES
from src.utils import reduce_mem_usage


def generate_features(features_df, transformers, label=None, encoder=None):
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
