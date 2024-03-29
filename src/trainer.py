import gc
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import wandb
from src.models import CBModel, LGBMModel


class Trainer:
    def __init__(
        self, cfg, id_col: str, tar_col: str, criterion, logger, features=None
    ):
        self.cfg = cfg
        self.features = features
        self.id_col = id_col
        self.tar_col = tar_col
        self.cv = StratifiedKFold(
            n_splits=cfg.data.n_splits, shuffle=True, random_state=cfg.data.seed
        )
        self.criterion = criterion
        self.logger = logger

        self.logger.info("Start Training")

    def _prepare_data(self, df):
        """
        prepare dataset for training
        ---------------------------------------------
        Parameter
        df: dataframe
            preprocessed data
        ---------------------------------------------
        Returns
        features, label, ids
        """

        if self.features is None:
            self.features = [
                f for f in df.columns if f not in [self.id_col, self.tar_col]
            ]

        self.cat_features_name = [
            c
            for c in df[self.features]
            .select_dtypes(include=["object", "category"])
            .columns
            if c.startswith("fe_")
        ]

        # Extract Features, label, Id
        data = df[self.features].values
        label = df[self.tar_col].values
        ids = df[self.id_col].values

        # logging
        self.logger.info(f"Training Data Shape: {data.shape}")

        return data, label, ids

    def _train_cv(self, data, label):
        """
        Train loop for Cross Validation
        """
        # init Model list
        self.models = []
        preds = np.zeros(len(label))
        oof_label = np.zeros(len(label))

        # Cross Validation Score
        for i, (trn_idx, val_idx) in enumerate(self.cv.split(data, label)):

            # Model init
            # LightGBM
            if self.cfg.train.model_type == "lgb":
                model = LGBMModel(dict(self.cfg.lgb))

            # CatBoost
            elif self.cfg.train.model_type == "catboost":
                model = CBModel(dict(self.cfg.catboost))
            else:
                raise (TypeError)

            # Train
            oof = model.train(
                x_train=data[trn_idx],
                y_train=label[trn_idx],
                x_val=data[val_idx],
                y_val=label[val_idx],
                features=self.features,
                cat_features=self.cat_features_name,
            )

            # Score
            score = self.criterion(label[val_idx], oof)

            # Logging
            wandb.log({"Fold Score": score}, step=i)
            print(f"Fold {i}  Score: {score:.3f}")
            preds[val_idx] = oof
            oof_label[val_idx] = label[val_idx]
            self.models.append(model)

            del model
            gc.collect()

        # All Fold Score
        oof_score = self.criterion(oof_label, preds)
        wandb.log({"Eval Score": oof_score})
        print(f"Eval Score: {oof_score:.3f}")
        auc = roc_auc_score(oof_label, preds)
        wandb.log({"Eval AUC": auc})
        print(f"Eval AUC: {auc:.3f}")

        return preds, self.models

    def _train_end(self, ids, preds, data=None, label=None):
        """
        End of Train loop per crossvalidation fold
        Logging and oof file
        """
        # Log params

        oof = pd.DataFrame({self.id_col: ids, self.tar_col: preds})

        oof = oof.sort_values(by=self.id_col)

        # Logging
        sub_name = "oof.csv"
        oof.to_csv(os.path.join(self.cfg.data.asset_dir, sub_name), index=False)
        time.sleep(5)
        wandb.save(os.path.join(self.cfg.data.asset_dir, sub_name))

        # Save Models
        sub_name = "models.pkl"
        with open(os.path.join(self.cfg.data.asset_dir, sub_name), "wb") as f:
            pickle.dump(self.models, f)
        time.sleep(5)
        wandb.save(os.path.join(self.cfg.data.asset_dir, sub_name))

        # Feature Importance
        feat_imp = np.zeros(len(self.features))
        for model in self.models:
            feat_imp += model.get_feature_importance(data, label)
        # Average Importance
        feat_imp /= len(self.models)

        feat_imp_df = pd.DataFrame({"feature": self.features, "importance": feat_imp})

        feat_imp_df = feat_imp_df.sort_values(
            by="importance", ascending=False
        ).reset_index(drop=True)

        sub_name = "feature_importance.csv"
        feat_imp_df.to_csv(os.path.join(self.cfg.data.asset_dir, sub_name), index=False)
        time.sleep(5)
        wandb.save(os.path.join(self.cfg.data.asset_dir, sub_name))

        wandb.log({"feature_importance": wandb.Table(dataframe=feat_imp_df)})

    def fit(self, df):
        data, label, ids = self._prepare_data(df)
        preds, models = self._train_cv(data, label)
        self._train_end(ids, preds, data, label)

        return models
