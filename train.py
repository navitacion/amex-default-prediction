import gc
import os

import yaml
import shutil
import wandb
import hydra
import numpy as np
from dotenv import load_dotenv
from logging import getLogger, config

from src.data import DataAsset
from src.models.lgbm import LGBMModel
from src.models.catboost import CatBoostModel, CatBoostAmexMetric
from src.trainer import Trainer
from src.inference import InferenceScoring
from src.utils import amex_metric
from src.features.base import generate_features


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
    wandb.init(project="amex-default-prediction")
    wandb.log(dict(cfg.data))

    with open('logging.yaml', 'r') as yml:
        logger_cfg = yaml.safe_load(yml)

    # 定義ファイルを使ったloggingの設定
    config.dictConfig(logger_cfg)

    # ロガーの取得
    logger = getLogger("Amex-PD Logger")

    # Load Dataset  --------------------------------------------
    asset = DataAsset(cfg, logger)
    org_features_df, label = asset.load_train_data()

    # Feature Extract  -----------------------------------------
    df, encoder = generate_features(org_features_df, label)
    del org_features_df, label
    gc.collect()

    # Model  ---------------------------------------------------
    # LightGBM
    wandb.log(dict(cfg.lgb))
    model = LGBMModel(dict(cfg.lgb))

    # Training  -------------------------------------------------
    trainer = Trainer(
        model, cfg,
        id_col='customer_ID',
        tar_col='target',
        features=None,
        criterion=amex_metric
    )

    models = trainer.fit(df)

    del df, model
    gc.collect()

    # Inference  -----------------------------------------------
    inferences = InferenceScoring(cfg, models, logger, encoder)
    inferences.run()


if __name__ == "__main__":
    main()
