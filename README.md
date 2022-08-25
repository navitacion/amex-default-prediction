# American Express - Default Prediction

Kaggle Competition Repogitory

https://www.kaggle.com/competitions/amex-default-prediction

## References

- [Target Encoder with smoothing](https://www.slideshare.net/0xdata/feature-engineering-83511751)
- [Integer columns in the data - here you go!](https://www.kaggle.com/competitions/amex-default-prediction/discussion/328514)

## Result

- Private Score: 0.805
- Rank: 2020th / 4937 (41%)

## Getting Started

Easy to do, only type command.

```commandline
docker-compose up --build -d
docker exec -it amex_env bash
```

## Solution

### Model Arch

- LightGBM (Dart)

### Submit Models

5 Random Seed Model Ensemble

- Private LB: 0.805
- Public LB: 0.798

### Cross Validation

- StratifiedKFold
	- 5 fold

### Preprocess

- Float to Integer [Discussion](https://www.kaggle.com/competitions/amex-default-prediction/discussion/328514)

### Feature Engineering

- Groupby Customer_ID (max, min, std, mean, last)
- DateDiff (days)
- Difference of previous record
- Rolling mean of previous record
- Count Transaction
- Count Null
- KMeans cluster ID & distance from center
- Target Encoder with Smoothing (Categorical & Integer)

#### Feature Selection

Using LightGBM Feature Importance(gain)

Top 2000 features only are used for model

## Model Training

To train Regression Model, execute the following command.

```commandline
python train.py
```

## Logging

- Wandb

## Helper Function

- train.py
	- For training model

- make_pickle.py
	- Preprocessing and make pickle file
	- Raw Data are huge, so it makes lighter for training