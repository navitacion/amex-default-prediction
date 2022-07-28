import numpy as np
import pandas as pd


class FrequencyEncoder:
    def __init__(self, feats=None):
        self.feats = feats

    def _prep(self, df):
        self.freq_dict = {}

        for c in self.feats:
            # colごとの値の個数を記録
            freq = df[c].value_counts() / len(df)
            self.freq_dict.update({c: freq})

    def transform(self, df, phase):

        if phase == "train":
            self._prep(df)

        res = df[["customer_ID"]]

        for c, v in self.freq_dict.items():
            col_name = f"fe_freq_enc_{c}"
            res[col_name] = df[c].map(v)
            res[col_name] = res[col_name].astype(float)

        return res

    def __call__(self, df, phase):
        return self.transform(df, phase)


class TargetEncoder:
    def __init__(self, feats=None, k=3, f=0.25):
        self.feats = feats
        self.k = k
        self.f = f

    def _prep(self, df):
        self.enc_dict = {}

        for c in self.feats:
            # colごとのTarget Encoding
            group = df.groupby(c)["target"].agg(["mean", "count"])
            group = pd.DataFrame(group).reset_index()

            group["dataset_mean"] = df["target"].mean()

            # Smoothing
            # Ref: https://www.slideshare.net/0xdata/feature-engineering-83511751
            group["lambda"] = group["count"].apply(
                lambda x: 1 / (1 + np.exp(-(x - self.k) / self.f))
            )
            group["target_enc"] = (
                group["lambda"] * group["mean"]
                + (1 - group["lambda"]) * group["dataset_mean"]
            )

            enc_dict = {k: v for k, v in zip(group[c], group["target_enc"])}

            self.enc_dict.update({c: enc_dict})

    def transform(self, df, phase):
        if phase == "train":
            self._prep(df)

        res = df[["customer_ID"]]

        for c, v in self.enc_dict.items():
            col_name = f"fe_tar_enc_smoothing_{c}"
            res[col_name] = df[c].map(v)
            res[col_name] = res[col_name].astype(float)

        return res

    def __call__(self, df, phase):
        return self.transform(df, phase)
