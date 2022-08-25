from pathlib import Path

import pandas as pd

paths = [str(p) for p in Path("./ensembles").glob("**/*.csv")]

print(paths)

weights = [0.3, 0.3, 0.4]

sub = pd.read_csv("./input/sample_submission.csv")
sub = sub.sort_values(by="customer_ID").reset_index(drop=True)
sub["prediction"] = 0

for path, w in zip(paths, weights):
    _df = pd.read_csv(path)
    _df = _df.sort_values(by="customer_ID").reset_index(drop=True)

    sub["prediction"] += _df["prediction"] * w

sub.to_csv("ensemble.csv", index=False)
