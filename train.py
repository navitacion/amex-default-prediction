import os
import pickle

import yaml
import shutil
import wandb
import hydra
from dotenv import load_dotenv
from logging import getLogger, config

from src.data import DataAsset
from src.models.lgbm import LGBMModel
from src.trainer import Trainer
from src.inference import InferenceScoring
from src.utils import amex_metric


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
    wandb.log(dict(cfg.lgb))

    with open('logging.yaml', 'r') as yml:
        logger_cfg = yaml.safe_load(yml)

    # 定義ファイルを使ったloggingの設定
    config.dictConfig(logger_cfg)

    # ロガーの取得
    logger = getLogger("Amex-PD Logger")

    # Load Dataset  --------------------------------------------
    asset = DataAsset(cfg, logger)

    df = asset.load_train_data()

    # Model  ---------------------------------------------------
    model = LGBMModel(dict(cfg.lgb))

    trainer = Trainer(
        model, cfg,
        id_col='customer_ID',
        tar_col='target',
        features=None,
        criterion=amex_metric
    )

    models = trainer.fit(df)

    # Inference  -----------------------------------------------
    inferences = InferenceScoring(cfg, models, logger)
    inferences.run()


if __name__ == "__main__":
    main()
