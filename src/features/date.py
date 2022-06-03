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
    def __init__(self):
        pass

    def transform(self, df):
        df['S_2'] = pd.to_datetime(df['S_2'])

        df['tmp'] = df.groupby('customer_ID')['S_2'].diff()
        df['tmp'] = df['tmp'].apply(lambda x: x.days)

        group = df.groupby('customer_ID')['tmp'].agg(['max', 'min', 'mean']).reset_index()

        rename_dict = {k: f"fe_transaction_days_{k}" for k in ['max', 'min', 'mean']}
        group = group.rename(columns=rename_dict)

        return group

    def __call__(self, df):
        return self.transform(df)
