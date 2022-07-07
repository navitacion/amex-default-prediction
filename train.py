import gc
import os
import time
import yaml
import shutil
import wandb
import hydra
import warnings
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from logging import getLogger, config

from src.data import DataAsset
from src.models import LGBMModel, CBModel
from src.trainer import Trainer
from src.inference import InferenceScoring
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
from src.constant import CAT_FEATURES, DATE_FEATURES, DROP_FEATURES

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
    org_features_df, label = asset.load_train_data()

    # Drop Features
    org_features_df = org_features_df.drop(DROP_FEATURES, axis=1)

    # Feature Extract  -----------------------------------------
    cnt_features = [
        f for f in org_features_df.columns if f not in CAT_FEATURES + DATE_FEATURES + ['customer_ID']
    ]

    transformers = [
        GroupbyIDTransformer(cnt_features, aggs=['max', 'min', 'mean', 'std', 'last']),
        GroupbyIDTransformer(CAT_FEATURES, aggs=['last']),
        TransactionDays(aggs=['max', 'mean', 'std']),
        RecentDiff(cnt_features, interval=1),
        RecentDiff(cnt_features, interval=2),
        RecentDiff(cnt_features, interval=3),
        # RollingMean(cnt_features, window=3),
        RollingMean(cnt_features, window=6),
        RecentPayDateDiffBeforePay(),
        CountTransaction(),  # 特徴量重要度が0
        NullCountPerCustomer(cnt_features + CAT_FEATURES),
    ]

    # Split Trains for avoiding memory killed
    train_ids = org_features_df['customer_ID'].unique().tolist()

    def _split_array(_data: list, n_group: int):
        for i_chunk in range(n_group):
            yield _data[i_chunk * len(_data) // n_group:(i_chunk + 1) * len(_data) // n_group]

    df = []
    for target_ids in tqdm(_split_array(train_ids, n_group=cfg.train.chunk_size), total=cfg.train.chunk_size):
        tmp = org_features_df[org_features_df['customer_ID'].isin(target_ids)].reset_index(drop=True)

        df.append(generate_features(tmp, transformers, logger, phase='train'))

    df = pd.concat(df, axis=0, ignore_index=True)

    df = pd.merge(df, label, on='customer_ID', how='left')

    del org_features_df, label
    gc.collect()

    # Model  ---------------------------------------------------
    # LightGBM
    if cfg.train.model_type == 'lgb':
        wandb.config.update(dict(cfg.lgb))
        model = LGBMModel(dict(cfg.lgb))

    # CatBoost
    elif cfg.train.model_type == 'catboost':
        wandb.config.update(dict(cfg.catboost))
        # Get Category
        cat_features = [
            c for c in df.select_dtypes(include=['object', 'category']).columns if c.startswith('fe_')
        ]
        model = CBModel(dict(cfg.catboost), cat_features)

    else:
        raise (TypeError)

    # Training  -------------------------------------------------
    logger.info(f'Train {cfg.train.model_type} Model')
    trainer = Trainer(
        model, cfg,
        id_col='customer_ID',
        tar_col='target',
        features=None,
        criterion=amex_metric
    )

    models = trainer.fit(df)

    del df, model, trainer
    gc.collect()

    # Inference  -----------------------------------------------
    if not cfg.debug:
        inferences = InferenceScoring(cfg, models, logger, transformers)
        inferences.run()

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
