from abc import ABCMeta, abstractmethod


# Basis -----------------------------------------------------------------------------------------------
class BaseModel(metaclass=ABCMeta):
    def __init__(self, params):
        self.params = params

    @abstractmethod
    def train(self, x_train, y_train, x_val, y_val, features, cat_features):
        raise NotImplementedError

    @abstractmethod
    def predict(self, x_test):
        raise NotImplementedError

    def get_feature_importance(self, features, label):
        pass
