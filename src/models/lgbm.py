import lightgbm as lgb
from wandb.lightgbm import log_summary

from src.models.base import BaseModel
from src.utils import amex_metric


# LightGBM -----------------------------------------------------------------------------------------------
# Custom Metric
def lgbm_amex_metric(preds, data):
    # 正解ラベル
    y_true = data.get_label()
    score = amex_metric(y_true, preds)

    return "amex_metric", score, True


class LGBMModel(BaseModel):
    def __init__(self, params):
        super(LGBMModel, self).__init__(params)

    def train(self, x_train, y_train, x_val, y_val, features, cat_features):
        self.model = None
        self.features = features
        train_data = lgb.Dataset(x_train, label=y_train)
        valid_data = lgb.Dataset(x_val, label=y_val, reference=train_data)

        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=[valid_data, train_data],
            valid_names=["eval", "train"],
            feature_name=features,
            feval=lgbm_amex_metric,
            verbose_eval=500,
        )

        log_summary(self.model, save_model_checkpoint=True)

        oof = self.model.predict(x_val, num_iteration=self.model.best_iteration)

        return oof

    def predict(self, x_test):
        pred = self.model.predict(x_test, num_iteration=self.model.best_iteration)
        return pred

    def get_feature_importance(self, features, label):
        return self.model.feature_importance(importance_type="gain")
