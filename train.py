import gc
import os
import pickle
import shutil
import time
import warnings
from logging import config, getLogger
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

import wandb
from src.constant import CAT_FEATURES, DATE_FEATURES, DROP_FEATURES
from src.data import DataAsset
from src.features.base import generate_features
from src.features.cluster import KmeansCluster, SVDExecuter
from src.features.date import (
    CountTransaction,
    RecentDiff,
    RecentPayDateDiffBeforePay,
    RollingMean,
    TransactionDays,
    TransactionRollingDays,
)
from src.features.encoding import TargetEncoder
from src.features.groupby import (
    GroupbyIDLastDiffTransformer,
    GroupbyIDLastDivTransformer,
    GroupbyIDTransformer,
    NullCountPerCustomer,
)
from src.trainer import Trainer
from src.utils import amex_metric, feature_selection_lgb, reduce_mem_usage

pd.options.display.max_rows = None
pd.options.display.max_columns = None


@hydra.main(config_path=".", config_name="config.yaml")
def main(cfg):
    cur_dir = hydra.utils.get_original_cwd()
    os.chdir(cur_dir)

    try:
        # Remove checkpoint folder
        shutil.rmtree(cfg.data.asset_dir)
    except FileNotFoundError:
        pass

    os.makedirs(cfg.data.asset_dir, exist_ok=True)

    # Overload Categorical Features from config
    cat_features = [c for c in CAT_FEATURES if c not in DROP_FEATURES]

    # Logger  --------------------------------------------------
    load_dotenv(".env")
    wandb.login(key=os.environ["WANDB_KEY"])
    wandb.init(project="amex-default-prediction", reinit=True)
    wandb.config.update(dict(cfg.data))
    wandb.config.update(dict(cfg.train))

    # Log Code
    wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))

    with open("logging.yaml", "r") as yml:
        logger_cfg = yaml.safe_load(yml)

    # 定義ファイルを使ったloggingの設定
    config.dictConfig(logger_cfg)

    # ロガーの取得
    logger = getLogger("Amex-PD Logger")

    # Load Dataset  --------------------------------------------
    asset = DataAsset(cfg, logger)

    # Feature Extract  -----------------------------------------
    t = pd.read_csv(Path(cfg.data.data_dir).joinpath("train_data.csv"), nrows=3)

    cnt_features = [
        f
        for f in t.columns
        if f not in cat_features + DATE_FEATURES + DROP_FEATURES + ["customer_ID"]
    ]

    del t
    gc.collect()

    transformers = [
        GroupbyIDTransformer(cnt_features, aggs=["max", "min", "std", "mean", "last"]),
        GroupbyIDLastDiffTransformer(
            cnt_features, aggs=["min", "max", "std", "mean", "first"]
        ),
        GroupbyIDLastDivTransformer(
            cnt_features, aggs=["min", "max", "std", "mean", "first"]
        ),
        GroupbyIDTransformer(cat_features, aggs=["last"]),
        TransactionDays(aggs=["max", "min", "mean", "std"]),
        TransactionRollingDays(aggs=["max", "min", "mean", "std"], window=2),
        TransactionRollingDays(aggs=["max", "min", "mean", "std"], window=3),
        TransactionRollingDays(aggs=["max", "min", "mean", "std"], window=4),
        RecentDiff(cnt_features, interval=1),
        RollingMean(cnt_features, window=2),
        RecentPayDateDiffBeforePay(recent_term=1),
        RecentPayDateDiffBeforePay(recent_term=2),
        RecentPayDateDiffBeforePay(recent_term=3),
        CountTransaction(),  # 特徴量重要度が0
        NullCountPerCustomer(cnt_features + cat_features),
    ]

    # Model  ---------------------------------------------------
    # LightGBM
    if cfg.train.model_type == "lgb":
        cfg.lgb.random_state = cfg.data.seed
        wandb.config.update(dict(cfg.lgb))

    # CatBoost
    elif cfg.train.model_type == "catboost":
        cfg.catboost.random_state = cfg.data.seed
        wandb.config.update(dict(cfg.catboost))
    else:
        raise (TypeError)

    # Feature Engineering  -------------------------------------------------
    logger.info(f"Train {cfg.train.model_type} Model")

    # Calc Features
    dfs = []
    for df in tqdm(asset.train_generator(), total=cfg.train.chunk_size):
        dfs.append(
            generate_features(df, transformers, cat_features, logger, phase="train")
        )
    df = pd.concat(dfs, axis=0, ignore_index=True)

    del dfs
    gc.collect()

    # Clustering
    # KMeans  ------------
    logger.info("Kmeans Features")
    kmeans = []
    for s in ["B", "D", "P", "S", "R"]:

        feats = [c for c in df.columns if f"key_ID_{s}_" in c]

        if len(feats) < cfg.prep.kmean_n_cluster:
            continue
        else:
            kmeans_prep = KmeansCluster(
                feats=feats,
                n_clusters=cfg.prep.kmean_n_cluster,
                seed=cfg.data.seed,
                suffix=s,
            )
            res = kmeans_prep(df, phase="train")
            df = pd.merge(df, res, on="customer_ID")
            kmeans.append(kmeans_prep)

    logger.info(f"Data Shape {df.shape}")

    # 次元圧縮  ------------
    # SVD
    logger.info("SVD Features")
    svds = []

    for s in ["B", "D", "P", "S", "R"]:
        feats = [c for c in df.columns if f"key_ID_{s}_" in c]

        if len(feats) < cfg.prep.svd_n_components:
            continue
        else:
            svd_prep = SVDExecuter(
                feats=feats,
                n_components=cfg.prep.svd_n_components,
                seed=cfg.data.seed,
                suffix=s,
            )
            res = svd_prep(df, phase="train")
            df = pd.merge(df, res, on="customer_ID")
            svds.append(svd_prep)

    logger.info(f"Data Shape {df.shape}")

    # Add label
    label = pd.read_csv(Path(cfg.data.data_dir).joinpath("train_labels.csv"))
    df = pd.merge(df, label, on="customer_ID", how="left")

    # Encodding  ---------------------------------------------
    # categoryとint型の変数を対象
    tar_cols = df.select_dtypes(include=[np.int8, np.int16, "category"]).columns

    # Target Encoding
    logger.info("Target Encoding Features")
    tar_enc = TargetEncoder(tar_cols)
    res = tar_enc(df, phase="train")
    df = pd.merge(df, res, on="customer_ID")
    logger.info(f"Data Shape {df.shape}")

    df = reduce_mem_usage(df)

    del label, res
    gc.collect()

    # Save as pickle
    filename = os.path.join(cfg.data.asset_dir, "train.pkl")
    with open(filename, "wb") as f:
        pickle.dump(df, f)
    wandb.save(filename)

    # Training  ----------------------------------------------------------
    # Feature Selection by LightGBM
    logger.info("Feature Selection")
    input_features = feature_selection_lgb(
        df, sample_frac=0.3, num_features=2000, seed=cfg.data.seed
    )
    logger.info(f"Using Feature Num: {len(input_features)}")

    # Set Trainer Class
    trainer = Trainer(
        cfg,
        id_col="customer_ID",
        tar_col="target",
        criterion=amex_metric,
        features=input_features,
        logger=logger,
    )

    # Train
    models = trainer.fit(df)

    del df, trainer
    gc.collect()

    # Inference  -----------------------------------------------
    if not cfg.debug:
        ids = []
        preds = []

        for df, target_ids in tqdm(
            asset.test_generator(), total=cfg.inference.chunk_size
        ):

            # Feature Engineering  ------------------------------
            df = generate_features(
                df, transformers, cat_features, logger=None, phase="predict"
            )

            # Kmeans
            for kmean in kmeans:
                res = kmean(df, phase="predict")
                df = pd.merge(df, res, on="customer_ID")

            # SVD
            for svd in svds:
                res = svd(df, phase="predict")
                df = pd.merge(df, res, on="customer_ID")

            # Encoding
            res = tar_enc(df, phase="predict")
            df = pd.merge(df, res, on="customer_ID")

            # Inference  -----------------------------------------
            pred = np.zeros(len(df))
            for model in models:
                pred += model.predict(df[model.features])

            # Avg
            pred /= len(models)

            ids.extend(target_ids)
            preds.extend(pred)

        res = pd.DataFrame({"customer_ID": ids, "prediction": preds})

        # Save to wandb
        sub_name = "submission.csv"
        res.to_csv(os.path.join(cfg.data.asset_dir, sub_name), index=False)
        wandb.save(os.path.join(cfg.data.asset_dir, sub_name))

    wandb.finish()
    time.sleep(3)

    # Remove checkpoint folder
    shutil.rmtree(cfg.data.asset_dir)
    shutil.rmtree("./wandb")

    # Clear Cache
    del transformers, models
    gc.collect()


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()
