import gc

import pandas as pd

from src.utils import reduce_mem_usage


def generate_features(
    features_df, transformers, cat_features, logger=None, phase="train"
):
    if logger is not None:
        logger.info("generate features")

    for c in cat_features:
        features_df[c] = features_df[c].astype("category")

    # Sort ID, S_2
    features_df = features_df.sort_values(by=["customer_ID", "S_2"])

    df = pd.DataFrame()

    for i, transformer in enumerate(transformers):
        if logger is not None:
            logger.info(f"Execute Feature {transformer.__class__.__name__}")

        _feats = transformer(features_df, phase=phase)

        _feats = reduce_mem_usage(_feats)

        if i == 0:
            df = _feats.copy()
        else:
            df = pd.merge(df, _feats, on=["customer_ID"], how="left")

        del _feats
        gc.collect()

        if logger is not None:
            logger.info(f"Data Shape {df.shape}")

    df = reduce_mem_usage(df)

    return df
