import gc
import pickle
from tqdm import tqdm
from pathlib import Path
import pandas as pd

from src.utils import reduce_mem_usage


def make_pickle(data_dir: str, _type: str = 'train', reduce_mem: bool = True):
    data_dir = Path(data_dir)

    # make Pickle file per feature type
    _df = pd.read_csv(data_dir.joinpath(f'{_type}_data.csv'), nrows=2)

    # S_2 looks like transaction date
    # PK: customer_ID + S_2
    for s in ['D', 'S', 'P', 'B', 'R']:
        use_cols = ['customer_ID', 'S_2'] + [c for c in _df.columns if c.startswith(s)]
        use_cols = list(set(use_cols))

        train = pd.read_csv(data_dir.joinpath(f'{_type}_data.csv'), chunksize=200000, usecols=use_cols)

        dfs = []
        for tmp in train:
            tmp = reduce_mem_usage(tmp) if reduce_mem else tmp
            dfs.append(tmp)

        # dfs = [df for df in train]
        dfs = pd.concat(dfs, axis=0, ignore_index=True)

        output_path = data_dir.joinpath(f'{_type}_data_{s}.pkl')

        with open(output_path, 'wb') as f:
            pickle.dump(dfs, f)

        del dfs
        gc.collect()

        print(f'Feature Type: {s} saved as pickle')

    return None


def make_test_label_pickle(data_dir: str):
    data_dir = Path(data_dir)

    # make Pickle file unique customer_ID contained Test Data
    test_customer_id = pd.read_csv(data_dir.joinpath('test_data.csv'), usecols=['customer_ID'])
    test_customer_id = test_customer_id.drop_duplicates()

    print(test_customer_id.shape)

    output_path = data_dir.joinpath(f'test_customer_ids.pkl')

    with open(output_path, 'wb') as f:
        pickle.dump(test_customer_id, f)

    print(f'Unique customer_ID List (Test) saved as pickle')


if __name__ == "__main__":
    data_dir = './input'

    print('Preparing Dataset')
    make_pickle(data_dir, _type='train', reduce_mem=True)
    make_test_label_pickle(data_dir)
    make_pickle(data_dir, _type='test', reduce_mem=True)
