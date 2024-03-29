import gc
import pickle
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.preprocessing import (
    floorify_random_noise_b,
    floorify_random_noise_d,
    floorify_random_noise_r,
    floorify_random_noise_s,
)
from src.utils import reduce_mem_usage


def make_test_label_pickle(data_dir: str):
    data_dir = Path(data_dir)

    # make Pickle file unique customer_ID contained Test Data
    test_customer_id = pd.read_csv(
        data_dir.joinpath("test_data.csv"), usecols=["customer_ID"]
    )
    test_customer_id = test_customer_id.drop_duplicates()

    print(test_customer_id.shape)

    output_path = data_dir.joinpath(f"test_customer_ids.pkl")

    with open(output_path, "wb") as f:
        pickle.dump(test_customer_id, f)

    print(f"Unique customer_ID List (Test) saved as pickle")


def make_pickle_prep(data_dir: str, _type: str = "train", reduce_mem: bool = True):
    """
    容量が大きいため、特徴量タイプごとにファイルを分ける
    """
    print(f"Make Pickle File {_type}")
    data_dir = Path(data_dir)

    # make Pickle file per feature type
    _df = pd.read_csv(data_dir.joinpath(f"{_type}_data.csv"), nrows=2)

    # S_2 looks like transaction date
    # PK: customer_ID + S_2
    for s in tqdm(["D", "S", "P", "B", "R"], total=5):
        use_cols = ["customer_ID", "S_2"] + [c for c in _df.columns if c.startswith(s)]
        use_cols = list(set(use_cols))

        test = pd.read_csv(
            data_dir.joinpath(f"{_type}_data.csv"), chunksize=200000, usecols=use_cols
        )

        dfs = []
        for tmp in test:
            tmp = reduce_mem_usage(tmp) if reduce_mem else tmp
            dfs.append(tmp)

        dfs = pd.concat(dfs, axis=0, ignore_index=True)

        if s == "D":
            dfs = floorify_random_noise_d(dfs)
        elif s == "S":
            dfs = floorify_random_noise_s(dfs)
        elif s == "B":
            dfs = floorify_random_noise_b(dfs)
        elif s == "R":
            dfs = floorify_random_noise_r(dfs)
        else:
            pass

        output_path = data_dir.joinpath(f"{_type}_data_prep_{s}.pkl")

        with open(output_path, "wb") as f:
            pickle.dump(dfs, f)

        del dfs
        gc.collect()

    return None


if __name__ == "__main__":
    data_dir = "./input"

    print("Preparing Dataset")
    make_pickle_prep(data_dir, _type="train", reduce_mem=True)
    make_pickle_prep(data_dir, _type="test", reduce_mem=True)
    make_test_label_pickle(data_dir)
