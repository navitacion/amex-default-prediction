import gc
import os

import yaml
import shutil
import wandb
import hydra
from dotenv import load_dotenv
from logging import getLogger, config

from src.data import DataAsset
from src.models import LGBMModel, CBModel
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
    if cfg.train.model_type == 'lgb':
        wandb.log(dict(cfg.lgb))
        model = LGBMModel(dict(cfg.lgb))

    # CatBoost
    elif cfg.train.model_type == 'catboost':
        wandb.log(dict(cfg.catboost))
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

    del df, model
    gc.collect()

    # Inference  -----------------------------------------------
    inferences = InferenceScoring(cfg, models, logger, encoder)
    inferences.run()


if __name__ == "__main__":
    main()
