#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

import xgboost as xgb
import mlflow

# Configure MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment-1")

models_folder = Path("models")
models_folder.mkdir(exist_ok=True)

def read_dataframe(year, month, nrows=100_000):
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
    cols = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID", "trip_distance"]
    df = pd.read_parquet(url, columns=cols)
    df = df.sample(n=nrows, random_state=42)

    df["duration"] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
    df = df[(df["duration"] >= 1) & (df["duration"] <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    return df

def create_X(df, dv=None):
    categorical = ["PULocationID", "DOLocationID"]
    numerical = ["trip_distance"]
    dicts = df[categorical + numerical].to_dict(orient="records")

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv

def train_model(X_train, y_train, X_val, y_val, dv):
    with mlflow.start_run() as run:
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            "learning_rate": 0.21791931703206877,
            "max_depth": 11,
            "min_child_weight": 17.72218118291325,
            "objective": "reg:squarederror",  # fixed deprecation warning
            "reg_alpha": 0.35833596417752367,
            "reg_lambda": 0.09446440043442567,
            "seed": 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        preprocessor_path = models_folder / "preprocessor.b"
        with open(preprocessor_path, "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact(str(preprocessor_path), artifact_path="preprocessor")
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        return run.info.run_id

def run(year, month):
    df_train = read_dataframe(year=year, month=month)

    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    df_val = read_dataframe(year=next_year, month=next_month)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    print(f"MLflow run_id: {run_id}")
    return run_id

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    args = parser.parse_args()

    run_id = run(year=args.year, month=args.month)

    if run_id:
        with open("run_id.txt", "w") as f:
            f.write(run_id)
