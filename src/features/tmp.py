import pandas as pd


def tmp_features(df: pd.DataFrame):
    # PK: customer_ID + S_2 -> customer_ID

    # customer_IDごとの最近のレコードを取ってくる
    max_s_2 = df.groupby('customer_ID')['S_2'].max().reset_index()

    trans_df = pd.merge(max_s_2, df, on=['customer_ID', 'S_2'], how='left')

    del trans_df['S_2']

    ids = trans_df['customer_ID'].values

    trans_df = trans_df.select_dtypes(exclude=['object'])
    trans_df['customer_ID'] = ids

    return trans_df
