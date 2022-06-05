import pandas as pd


class CountTransaction:
    def __init__(self):
        pass

    def transform(self, df):
        # customer_IDごとの最近のレコードを取ってくる
        trans_df = df.groupby('customer_ID')['S_2'].nunique().reset_index()
        trans_df = trans_df.rename(columns={'S_2': 'fe_transaction_num'})

        return trans_df

    def __call__(self, df):
        return self.transform(df)


class TransactionDays:
    def __init__(self, aggs: list):
        """customer_IDごとの取引日(S_2)のStats"""
        self.aggs = aggs

    def transform(self, df):
        df['S_2'] = pd.to_datetime(df['S_2'])

        df['tmp'] = df.groupby('customer_ID')['S_2'].diff()
        df['tmp'] = df['tmp'].apply(lambda x: x.days)

        group = df.groupby('customer_ID')['tmp'].agg(self.aggs).reset_index()

        rename_dict = {k: f"fe_transaction_days_{k}" for k in self.aggs}
        group = group.rename(columns=rename_dict)

        return group

    def __call__(self, df):
        return self.transform(df)


class P2Increase:
    def __init__(self, aggs):
        self.aggs = aggs

    def transform(self, df):
        # customer_IDごとの最近のレコードを取ってくる
        df['tmp'] = df.groupby('customer_ID')['P_2'].diff()
        group = df.groupby('customer_ID')['tmp'].agg(self.aggs).reset_index()

        rename_dict = {k: f"fe_p_2_diff_{k}" for k in self.aggs}
        group = group.rename(columns=rename_dict)

        return group

    def __call__(self, df):
        return self.transform(df)

# TODO: 最新月からの急激な変化を検知したい
# class RecentDiff:
#     def __init__(self, recent_from_days: list):
#         self.recent_from_days = recent_from_days
#
#     def transform(self, df):
#         # customer_IDごとの最近のレコードを取ってくる
#         df['tmp'] = df.groupby('customer_ID')['P_2'].diff()
#         group = df.groupby('customer_ID')['tmp'].agg(self.aggs).reset_index()
#
#         rename_dict = {k: f"fe_p_2_diff_{k}" for k in self.aggs}
#         group = group.rename(columns=rename_dict)
#
#         return group
#
#     def __call__(self, df):
#         return self.transform(df)
