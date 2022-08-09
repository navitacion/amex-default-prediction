import gc

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, Ridge


def reduce_mem_usage(df):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    with np.errstate(invalid="ignore"):
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == "int":
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif (
                        c_min > np.iinfo(np.int16).min
                        and c_max < np.iinfo(np.int16).max
                    ):
                        df[col] = df[col].astype(np.int16)
                    elif (
                        c_min > np.iinfo(np.int32).min
                        and c_max < np.iinfo(np.int32).max
                    ):
                        df[col] = df[col].astype(np.int32)
                    elif (
                        c_min > np.iinfo(np.int64).min
                        and c_max < np.iinfo(np.int64).max
                    ):
                        df[col] = df[col].astype(np.int64)
                else:
                    # df[col] = df[col].astype(np.float32)
                    if (
                        c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max
                    ):
                        df[col] = df[col].astype(np.float16)
                    elif (
                        c_min > np.finfo(np.float32).min
                        and c_max < np.finfo(np.float32).max
                    ):
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float32)

    return df


# Ref: https://www.kaggle.com/competitions/amex-default-prediction/discussion/327162
def amex_metric(y_true: np.array, y_pred: np.array) -> float:
    # ndarray -> pd.DataFrame
    y_true = pd.DataFrame({"target": y_true})

    y_pred = pd.DataFrame({"prediction": y_pred})

    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = pd.concat([y_true, y_pred], axis="columns").sort_values(
            "prediction", ascending=False
        )
        df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
        four_pct_cutoff = int(0.04 * df["weight"].sum())
        df["weight_cumsum"] = df["weight"].cumsum()
        df_cutoff = df.loc[df["weight_cumsum"] <= four_pct_cutoff]
        return (df_cutoff["target"] == 1).sum() / (df["target"] == 1).sum()

    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = pd.concat([y_true, y_pred], axis="columns").sort_values(
            "prediction", ascending=False
        )
        df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
        df["random"] = (df["weight"] / df["weight"].sum()).cumsum()
        total_pos = (df["target"] * df["weight"]).sum()
        df["cum_pos_found"] = (df["target"] * df["weight"]).cumsum()
        df["lorentz"] = df["cum_pos_found"] / total_pos
        df["gini"] = (df["lorentz"] - df["random"]) * df["weight"]
        return df["gini"].sum()

    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={"target": "prediction"})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)

    return 0.5 * (g + d)


def feature_selection(df, sample_frac=1.0, seed=0):
    # Null Rate
    nulls = df.isnull().sum() / len(df)

    nulls = nulls[nulls < 0.5].index.tolist()
    nulls = [c for c in nulls if c.startswith("fe_")]

    # 欠損値がない場合
    # Ridge回帰による特徴量選定
    tmp = df.dropna(axis=1, how="any")
    feats = [c for c in tmp.columns if c.startswith("fe_")]

    sampled = df.sample(frac=sample_frac, random_state=seed)

    clf = Lasso()

    sfm = SelectFromModel(clf)
    sfm.fit(sampled[feats], sampled["target"])

    del sampled
    gc.collect()

    selected_feats = np.array(feats)[sfm.get_support()].tolist()

    selected_feats = selected_feats + nulls

    selected_feats = list(set(selected_feats))

    return selected_feats
