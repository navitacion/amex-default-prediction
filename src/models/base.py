from abc import ABCMeta, abstractmethod
import lightgbm as lgb
from wandb.lightgbm import wandb_callback, log_summary

from ..utils import amex_metric

# Basis -----------------------------------------------------------------------------------------------
class BaseModel(metaclass=ABCMeta):
    def __init__(self, params):
        self.params = params

    @abstractmethod
    def train(self, x_train, y_train, x_val, y_val):
        raise NotImplementedError

    @abstractmethod
    def predict(self, x_test):
        raise NotImplementedError

    def get_feature_importance(self):
        pass



# LightGBM -----------------------------------------------------------------------------------------------
# Custom Metric
def lgbm_amex_metric(preds, data):
    # 正解ラベル
    y_true = data.get_label()
    score = amex_metric(y_true, preds)

    return 'amex_metric', score, True


class LGBMModel(BaseModel):
    def __init__(self, params):
        super(LGBMModel, self).__init__(params)

    def train(self, x_train, y_train, x_val, y_val, feature_name='auto'):
        train_data = lgb.Dataset(x_train, label=y_train)
        valid_data = lgb.Dataset(x_val, label=y_val, reference=train_data)

        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=[valid_data, train_data],
            valid_names=['eval', 'train'],
            feature_name=feature_name,
            # callbacks=[wandb_callback()],
            feval=lgbm_amex_metric,
        )

        log_summary(self.model, save_model_checkpoint=True)

        oof = self.model.predict(x_val, num_iteration=self.model.best_iteration)

        return oof


    def predict(self, x_test):
        pred = self.model.predict(x_test, num_iteration=self.model.best_iteration)
        return pred

    def get_feature_importance(self):
        return self.model.feature_importance()


