import gc
import os
import pickle
import time
import yaml
import shutil
import wandb
import hydra
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from logging import getLogger, config

from src.data import DataAsset
from src.models import LGBMModel, CBModel
from src.trainer import Trainer
from src.utils import amex_metric
from src.features.base import generate_features
from src.features.groupby import GroupbyIDTransformer, NullCountPerCustomer
from src.features.date import (
    CountTransaction,
    TransactionDays,
    RecentDiff,
    RollingMean,
    RecentPayDateDiffBeforePay
)
from src.features.cluster import KmeansCluster
from src.features.encoding import FrequencyEncoder, TargetEncoder
from src.constant import CAT_FEATURES, DATE_FEATURES, DROP_FEATURES
from src.utils import reduce_mem_usage

pd.options.display.max_rows = None
pd.options.display.max_columns = None


@hydra.main(config_path=".", config_name="config.yaml")
def main(cfg):
    cur_dir = hydra.utils.get_original_cwd()
    os.chdir(cur_dir)

    try:
        # Remove checkpoint folder
        shutil.rmtree(cfg.data.asset_dir)
    except:
        pass

    os.makedirs(cfg.data.asset_dir, exist_ok=True)

    # Logger  --------------------------------------------------
    load_dotenv('.env')
    wandb.login(key=os.environ['WANDB_KEY'])
    wandb.init(project="amex-default-prediction", reinit=True)
    wandb.config.update(dict(cfg.data))
    wandb.config.update(dict(cfg.train))

    with open('logging.yaml', 'r') as yml:
        logger_cfg = yaml.safe_load(yml)

    # 定義ファイルを使ったloggingの設定
    config.dictConfig(logger_cfg)

    # ロガーの取得
    logger = getLogger("Amex-PD Logger")

    # Load Dataset  --------------------------------------------
    asset = DataAsset(cfg, logger)

    # Feature Extract  -----------------------------------------
    t = pd.read_csv(Path(cfg.data.data_dir).joinpath('train_data.csv'), nrows=3)

    cnt_features = [
        f for f in t.columns if f not in CAT_FEATURES + DATE_FEATURES + DROP_FEATURES + ['customer_ID']
    ]

    transformers = [
        GroupbyIDTransformer(cnt_features, aggs=['max', 'min', 'mean', 'last']),
        GroupbyIDTransformer(CAT_FEATURES, aggs=['last']),
        TransactionDays(aggs=['max', 'mean', 'std']),
        RecentDiff(cnt_features, interval=1),
        RecentDiff(cnt_features, interval=2),
        # RecentDiff(cnt_features, interval=3),
        # RollingMean(cnt_features, window=3),
        RollingMean(cnt_features, window=6),
        RecentPayDateDiffBeforePay(),
        # CountTransaction(),  # 特徴量重要度が0
        # NullCountPerCustomer(cnt_features + CAT_FEATURES),
    ]

    # Model  ---------------------------------------------------
    # LightGBM
    if cfg.train.model_type == 'lgb':
        wandb.config.update(dict(cfg.lgb))
        model = LGBMModel(dict(cfg.lgb))

    # CatBoost
    elif cfg.train.model_type == 'catboost':
        wandb.config.update(dict(cfg.catboost))
        model = CBModel(dict(cfg.catboost))
    else:
        raise (TypeError)

    # Feature Engineering  -------------------------------------------------
    logger.info(f'Train {cfg.train.model_type} Model')

    # Calc Features
    dfs = []
    for df in tqdm(asset.train_generator(), total=cfg.train.chunk_size):
        dfs.append(generate_features(df, transformers, logger, phase='train'))
    df = pd.concat(dfs, axis=0, ignore_index=True)

    # TODO: Kmeans系の特徴量
    kmeans_prep = KmeansCluster(
        n_clusters=cfg.prep.kmean_n_cluster,
        seed=cfg.data.seed
    )
    res = kmeans_prep(df, phase='train')
    df = pd.merge(df, res, on='customer_ID')

    # Add label
    label = pd.read_csv(Path(cfg.data.data_dir).joinpath('train_labels.csv'))
    df = pd.merge(df, label, on='customer_ID', how='left')

    # TODO: encoding系
    # categoryとint型の変数を対象
    tar_cols = df.select_dtypes(include=[np.int8, np.int16, 'category']).columns
    freq_enc = FrequencyEncoder(tar_cols)
    res = freq_enc(df, phase='train')
    df = pd.merge(df, res, on='customer_ID')

    tar_enc = TargetEncoder(tar_cols)
    res = tar_enc(df, phase='train')
    df = pd.merge(df, res, on='customer_ID')

    logger.info(f'Train Data Shape: {df.shape}')
    df = reduce_mem_usage(df)
    print(df.dtypes)

    del label
    gc.collect()

    # Save as pickle
    filename = os.path.join(cfg.data.asset_dir, 'train.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(df, f)
    wandb.save(filename)

    # Training  ----------------------------------------------------------
    # Set Trainer Class
    trainer = Trainer(
        model, cfg,
        id_col='customer_ID',
        tar_col='target',
        criterion=amex_metric
    )

    # Train
    models = trainer.fit(df)

    del df, model, trainer
    gc.collect()

    # Inference  -----------------------------------------------
    if not cfg.debug:
        ids = []
        preds = []

        for df, target_ids in tqdm(asset.test_generator(), total=cfg.inference.chunk_size):

            # Feature Engineering  ------------------------------
            df = generate_features(df, transformers, logger=None, phase='predict')

            # TODO: KMeans系の特徴量
            res = kmeans_prep(df, phase='predict')
            df = pd.merge(df, res, on='customer_ID')

            # TODO: Encoding系の特徴量
            res = freq_enc(df, phase='predict')
            df = pd.merge(df, res, on='customer_ID')

            res = tar_enc(df, phase='predict')
            df = pd.merge(df, res, on='customer_ID')

            # Inference  -----------------------------------------
            pred = np.zeros(len(df))
            for model in models:
                pred += model.predict(df[model.features])

            # Avg
            pred /= len(models)

            ids.extend(target_ids)
            preds.extend(pred)

            print(len(ids))
            print(len(preds))

        res = pd.DataFrame({
            'customer_ID': ids,
            'prediction': preds
        })

        # Save to wandb
        sub_name = 'submission.csv'
        res.to_csv(os.path.join(cfg.data.asset_dir, sub_name), index=False)
        wandb.save(os.path.join(cfg.data.asset_dir, sub_name))

    wandb.finish()
    time.sleep(3)

    # Remove checkpoint folder
    shutil.rmtree(cfg.data.asset_dir)
    shutil.rmtree('./wandb')

    # Clear Cache
    del transformers, models
    gc.collect()


if __name__ == "__main__":
    warnings.simplefilter('ignore')
    main()
