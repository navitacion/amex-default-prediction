import gc

import numpy as np
import pandas as pd


class GroupbyIDTransformer:
    def __init__(self, feats, aggs: list):
        self.feats = feats
        self.aggs = aggs

    def _unique_customer_id(self, df):
        target_df = pd.DataFrame({"customer_ID": df["customer_ID"].unique()})

        return target_df

    def transform(self, df, phase):
        target = self._unique_customer_id(df)

        for agg in self.aggs:
            group = df.groupby("customer_ID")[self.feats].agg(agg).reset_index()
            rename_dict = {k: f"fe_group_{agg}_key_ID_{k}" for k in self.feats}
            group = group.rename(columns=rename_dict)

            target = pd.merge(target, group, on="customer_ID")

            del group
            gc.collect()

        return target

    def __call__(self, df, phase):
        return self.transform(df, phase)


class GroupbyIDFuncTransformer:
    def __init__(self, feats, funcs: dict):
        self.feats = feats
        self.funcs = funcs

    def _unique_customer_id(self, df):
        target_df = pd.DataFrame({"customer_ID": df["customer_ID"].unique()})

        return target_df

    def transform(self, df, phase):
        target = self._unique_customer_id(df)

        for n, func in self.funcs.items():
            group = df.groupby("customer_ID")[self.feats].apply(func).reset_index()
            rename_dict = {k: f"fe_group_{n}_key_ID_{k}" for k in self.feats}
            group = group.rename(columns=rename_dict)

            target = pd.merge(target, group, on="customer_ID")

            del group
            gc.collect()

        return target

    def __call__(self, df, phase):
        return self.transform(df, phase)


class NullCountPerCustomer:
    """
    ユーザーごとの特徴慮の欠損数

    """

    def __init__(self, feats: list):
        self.feats = feats

    def transform(self, df, phase):
        group = (
            df.groupby("customer_ID")[self.feats]
            .count()
            .rsub(df.groupby("customer_ID").size(), axis=0)
            .reset_index()
        )

        rename_dict = {c: f"fe_nullCount_{c}" for c in self.feats}
        group = group.rename(columns=rename_dict)

        for _, v in rename_dict.items():
            group[v] = group[v].astype(np.uint8)

        # すべて同じ値（欠損値がない）カラムは除外
        if phase == "train":
            drop_f = []
            for c in group.columns:
                if group[c].max() == group[c].min():
                    drop_f.append(c)

            group = group.drop(drop_f, axis=1)

        return group

    def __call__(self, df, phase):
        return self.transform(df, phase)
