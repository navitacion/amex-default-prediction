import os, pickle
import numpy as np
import pandas as pd
import wandb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


class Trainer:
    def __init__(self, model, cfg, id_col: str, tar_col: str, features, criterion):
        self.model = model
        self.cfg = cfg
        self.id_col = id_col
        self.tar_col = tar_col
        self.features = features
        self.cv = StratifiedKFold(n_splits=cfg.data.n_splits, shuffle=True, random_state=cfg.data.seed)
        self.criterion = criterion

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
            self.features = [f for f in df.columns if f not in [self.id_col, self.tar_col]]

        # Extract Features, label, Id
        features = df[self.features]
        label = df[self.tar_col].values
        ids = df[self.id_col].values

        return features, label, ids

    def _train_cv(self, features, label):
        """
        Train loop for Cross Validation
        """
        # init Model list
        self.models = []
        preds = np.zeros(len(label))
        oof_label = np.zeros(len(label))

        # Cross Validation Score
        for i, (trn_idx, val_idx) in enumerate(self.cv.split(features, label)):
            x_trn, y_trn = features.iloc[trn_idx], label[trn_idx]
            x_val, y_val = features.iloc[val_idx], label[val_idx]

            oof = self.model.train(x_trn, y_trn, x_val, y_val, features=self.features)

            # Score
            score = self.criterion(y_val, oof)

            # Logging
            wandb.log({'Fold Score': score}, step=i)
            print(f'Fold {i}  Score: {score:.3f}')
            preds[val_idx] = oof
            oof_label[val_idx] = y_val
            self.models.append(self.model)

        # All Fold Score
        oof_score = self.criterion(oof_label, preds)
        wandb.log({'Eval Score': oof_score})
        print(f'Eval Score: {oof_score:.3f}')
        auc = roc_auc_score(oof_label, preds)
        wandb.log({'Eval AUC': auc})
        print(f'Eval AUC: {auc:.3f}')

        return preds, self.models

    def _train_end(self, ids, preds):
        """
        End of Train loop per crossvalidation fold
        Logging and oof file
        """
        # Log params

        oof = pd.DataFrame({
            self.id_col: ids,
            self.tar_col: preds
        })

        oof = oof.sort_values(by=self.id_col)

        # Logging
        sub_name = 'oof.csv'
        oof.to_csv(os.path.join(self.cfg.data.asset_dir, sub_name), index=False)
        wandb.save(os.path.join(self.cfg.data.asset_dir, sub_name))

        wandb.log({
            'oof_pred': wandb.Table(dataframe=oof)
        })

        # Save Models
        sub_name = 'models.pkl'
        with open(os.path.join(self.cfg.data.asset_dir, sub_name), 'wb') as f:
            pickle.dump(self.models, f)
        wandb.save(os.path.join(self.cfg.data.asset_dir, sub_name))

        # Feature Importances
        try:
            feat_imp = np.zeros(len(self.features))
            for model in self.models:
                feat_imp += model.get_feature_importance()

            feat_imp /= len(self.models)

            feat_imp_df = pd.DataFrame({
                'feature': self.features,
                'importance': feat_imp
            })

            sub_name = 'feature_importance.csv'
            feat_imp_df.to_csv(os.path.join(self.cfg.data.asset_dir, sub_name), index=False)
            wandb.save(os.path.join(self.cfg.data.asset_dir, sub_name))

            wandb.log({
                'feature_importance': wandb.Table(dataframe=feat_imp_df)
            })

        except:
            pass

    def fit(self, df):
        features, label, ids = self._prepare_data(df)
        preds, models = self._train_cv(features, label)
        self._train_end(ids, preds)

        return models
