import os
import shutil
import wandb
import hydra
from dotenv import load_dotenv

from src.data import DataAsset
from src.models.base import LGBMModel
from src.trainer import Trainer
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

    # Load Dataset  --------------------------------------------
    dataasset = DataAsset(cfg)

    df = dataasset.load_train_data()

    # Model  ---------------------------------------------------
    model = LGBMModel(dict(cfg.lgb))

    trainer = Trainer(
        model, cfg,
        id_col='customer_ID',
        tar_col='target',
        features=None,
        criterion=amex_metric
    )

    trainer.fit(df)

    # Predict  -------------------------------------------------

    # test = dataasset.get_generator_loading_test(chunksize=200)

    # for t in test:
    #     print(t.shape)
    #     break


if __name__ == "__main__":
    main()
