import cudf
import numpy as np
import pandas as pd


class CountTransaction:
    def __init__(self):
        """
        customer_IDごとの取引回数
        """
        pass

    def transform(self, df, phase):
        # customer_IDごとの最近のレコードを取ってくる
        trans_df = df.groupby("customer_ID")["S_2"].nunique().reset_index()
        trans_df = trans_df.rename(columns={"S_2": "fe_transaction_num"})

        # 毎月取引している場合は貸し倒れが小さい
        trans_df["fe_is_transaction_num_over_13"] = trans_df[
            "fe_transaction_num"
        ].apply(lambda x: 1 if x == 13 else 0)

        trans_df["fe_transaction_num"] = trans_df["fe_transaction_num"].astype(np.uint8)
        trans_df["fe_is_transaction_num_over_13"] = trans_df[
            "fe_is_transaction_num_over_13"
        ].astype(np.uint8)

        return trans_df

    def __call__(self, df, phase):
        return self.transform(df, phase)


class TransactionDays:
    def __init__(self, aggs: list):
        """
        customer_IDごとの取引日(S_2)の日数のStats
        customer_IDごとに取引日の期間にフォーカスしたもの
        """
        self.aggs = aggs

    def transform(self, df, phase):
        df["S_2"] = pd.to_datetime(df["S_2"])

        df["tmp"] = df.groupby("customer_ID")["S_2"].diff()
        df["tmp"] = df["tmp"].apply(lambda x: x.days)

        group = df.groupby("customer_ID")["tmp"].agg(self.aggs).reset_index()

        rename_dict = {k: f"fe_transaction_days_{k}" for k in self.aggs}
        group = group.rename(columns=rename_dict)

        return group

    def __call__(self, df, phase):
        return self.transform(df, phase)


class RecentDiff:
    """
    直近の値の差分
    interval = 1  各特徴量の最近の値とその一つ前の値の差分

    """

    def __init__(self, feats: list, interval: int):
        self.feats = feats
        self.interval = interval

    def transform(self, df, phase):
        _df = df[["customer_ID", "S_2"] + self.feats].copy()

        _df = cudf.from_pandas(_df)

        for f in self.feats:
            # customer_IDごとに特徴量の差分を計算する
            _df[f"fe_recentDiff_interval_{self.interval}_{f}"] = _df.groupby(
                "customer_ID"
            )[f].diff(self.interval)

            del _df[f]

        _df = _df.to_pandas()
        # 最新の日付の差分を取る
        _df = _df.groupby("customer_ID").tail(1).reset_index(drop=True)

        # S_2を削除する
        del _df["S_2"]

        return _df

    def __call__(self, df, phase):
        return self.transform(df, phase)


class RollingMean:
    def __init__(self, feats: list, window: int):
        self.feats = feats
        self.window = window

    def transform(self, df, phase):
        _df = df[["customer_ID", "S_2"] + self.feats].copy()

        _df = cudf.from_pandas(_df)

        # 移動平均
        def func(x):
            n = 0
            for i in range(self.window):
                n += x.shift(i) / self.window
            return n

        for f in self.feats:
            # customer_IDごとに特徴量の差分を計算する
            _df[f"fe_rollingMean_window_{self.window}_{f}"] = _df.groupby(
                "customer_ID"
            )[f].pipe(func)

            del _df[f]

        _df = _df.to_pandas()
        # 最新の日付の差分を取る
        _df = _df.groupby("customer_ID").tail(1).reset_index(drop=True)

        # S_2を削除する
        del _df["S_2"]

        return _df

    def __call__(self, df, phase):
        return self.transform(df, phase)


class RecentPayDateDiffBeforePay:
    """
    最新の取引から直前までの取引までの差分

    """

    def __init__(self):
        pass

    def transform(self, df, phase):
        _df = df[["customer_ID", "S_2"]].copy()
        _df["S_2"] = pd.to_datetime(_df["S_2"])

        _df = cudf.from_pandas(_df)

        feat_name = "fe_recentPayDiffBeforePay"
        _df[feat_name] = _df.groupby("customer_ID")["S_2"].diff()

        _df = _df.to_pandas()

        _df[feat_name] = _df[feat_name].apply(lambda x: x.days)

        # 最新の日付の差分を取る
        _df = _df.groupby("customer_ID").tail(1).reset_index(drop=True)
        _df = _df[["customer_ID", feat_name]]

        return _df

    def __call__(self, df, phase):
        return self.transform(df, phase)


# TODO: 同じ年月における全ユーザーの平均値からの比率
