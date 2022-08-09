import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, EFstrType, Pool

from src.models.base import BaseModel


# https://www.kaggle.com/competitions/amex-default-prediction/discussion/328020
# https://www.kaggle.com/code/rohanrao/amex-competition-metric-implementations
# Custom Metric
class CBAmexMetric(object):
    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        return True

    def evaluate(self, y_pred, y_true, weight):
        y_pred = y_pred[0]
        indices = np.argsort(y_pred)[::-1]
        preds, target = y_pred[indices], y_true[indices]
        weight = 20.0 - target * 19.0
        cum_norm_weight = (weight / weight.sum()).cumsum()
        four_pct_mask = cum_norm_weight <= 0.04
        d = np.sum(target[four_pct_mask]) / np.sum(target)
        weighted_target = target * weight
        lorentz = (weighted_target / weighted_target.sum()).cumsum()
        gini = ((lorentz - cum_norm_weight) * weight).sum()
        n_pos = np.sum(target)
        n_neg = target.shape[0] - n_pos
        gini_max = 10 * n_neg * (n_pos + 20 * n_neg - 19) / (n_pos + 20 * n_neg)
        g = gini / gini_max
        return 0.5 * (g + d), 0


class CBModel(BaseModel):
    def __init__(self, params):
        super(CBModel, self).__init__(params)
        self.params["eval_metric"] = CBAmexMetric()

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        x_val: pd.DataFrame,
        y_val: pd.DataFrame,
        features: list,
        cat_features: list,
    ) -> pd.DataFrame:
        self.model = None
        self.features = features
        self.cat_features = cat_features

        train_data = Pool(x_train, y_train, cat_features=self.cat_features)
        val_data = Pool(x_val, y_val, cat_features=self.cat_features)

        self.model = CatBoostClassifier(**self.params)
        self.model.fit(train_data, eval_set=val_data, verbose=100)

        oof = self.model.predict(x_val, prediction_type="Probability")[:, 1]

        return oof

    def predict(self, x_test):
        pred = self.model.predict(x_test, prediction_type="Probability")[:, 1]
        return pred

    def get_feature_importance(self, features, label):
        # Ref: https://github.com/catboost/catboost/blob/master/catboost/python-package/catboost/core.py#L209-L223
        # Ref: https://catboost.ai/en/docs/concepts/cli-reference_fstr-calc#description
        features = pd.DataFrame(features, columns=self.features)
        features_pool = Pool(features, label, cat_features=self.cat_features)

        return self.model.get_feature_importance(
            features_pool, type=EFstrType.LossFunctionChange
        )
