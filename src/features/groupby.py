import pandas as pd


class GroupbyIDTransformer:
    def __init__(self, feats, aggs: list):
        self.feats = feats
        self.aggs = aggs

    def _unique_customer_id(self, df):
        target_df = pd.DataFrame({
            'customer_ID': df['customer_ID'].unique()
        })

        return target_df

    def transform(self, df):
        target = self._unique_customer_id(df)

        for agg in self.aggs:
            group = df.groupby('customer_ID')[self.feats].agg(agg).reset_index()
            rename_dict = {k: f"fe_group_{agg}_key_customer_ID_feat_{k}" for k in self.feats}
            group = group.rename(columns=rename_dict)

            target = pd.merge(target, group, on='customer_ID')

        return target

    def __call__(self, df):
        return self.transform(df)
