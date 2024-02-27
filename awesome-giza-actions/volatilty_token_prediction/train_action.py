import pandas as pd
from utils import (remove_columns_with_nulls_above_threshold,
                    daily_price_dateset_manipulation, 
                    apy_dateset_manipulation, 
                    tvl_dateset_manipulation,
                    dimensionality_reduction)
import numpy as np
import onnx
from giza_actions.task import task
from giza_actions.action import Action, action
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor
from hummingbird.ml import convert

import certifi
import os
os.environ['SSL_CERT_FILE'] = certifi.where()

@task(name='Join and postprocessing')
def loading_and_processing():
    """
    Loads, merges, and processes datasets for further analysis.
    
    Returns:
        pd.DataFrame: The processed dataframe after merging and cleaning.
    """
    df_main = daily_price_dateset_manipulation()
    apy_df = apy_dateset_manipulation()
    tvl_df = tvl_dateset_manipulation()

    # Merge dataframes on the 'date' column
    df_main = df_main.merge(tvl_df.to_pandas(), on="date", how="inner")
    df_main = df_main.merge(apy_df.to_pandas(), on="date", how="inner")

    # Remove columns with more than 5% null values
    df_main = remove_columns_with_nulls_above_threshold(df_main, 0.05)
    return df_main

@task(name='prepare dataset')
def prepare_dataset(df, test_n=60):
    """
    Prepares the dataset for training and testing by splitting and applying dimensionality reduction.
    
    Parameters:
        df (pd.DataFrame): The dataframe to prepare.
        test_n (int): Number of samples to use for the test set.
    
    Returns:
        tuple: A tuple containing the training and testing datasets (X_train, X_test, y_train, y_test).
    """
    X = df.drop(['WETH_future_vol', 'date'], axis=1)
    y = df['WETH_future_vol']

    # Split the data into training and testing sets
    X_train = X.iloc[:-test_n]
    X_test = X.iloc[-test_n:]
    y_train = y.iloc[:-test_n]
    y_test = y.iloc[-test_n:]

    # Apply dimensionality reduction on the training set and align test set accordingly
    X_train = dimensionality_reduction(X_train, y_train)
    X_test = X_test[X_train.columns]
    return X_train, X_test, y_train, y_test

@task(name='train model')
def train_model(X, y):
    """
    Trains a LightGBM regressor model using time series cross-validation.
    
    Parameters:
        X (pd.DataFrame): Feature matrix for training.
        y (pd.Series): Target variable.
    
    Returns:
        LGBMRegressor: The trained LightGBM model.
    """
    params = {
        'learning_rate': 0.005,
        'n_estimators': 1000,
        'early_stopping_rounds': 50, 
        'verbose': -1
    }

    tscv = TimeSeriesSplit(n_splits=5)
    optimal_rounds = []

    # Perform time series cross-validation to find the optimal number of boosting rounds
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        optimal_rounds.append(model.best_iteration_)

    optimal_rounds_avg = sum(optimal_rounds) / len(optimal_rounds)

    # Train the final model on the entire dataset with optimized parameters
    model_full = LGBMRegressor(learning_rate=0.005, 
                            n_estimators=int(optimal_rounds_avg * 1.1),
                            max_depth=6,
                            min_data_in_leaf=20,
                            num_leaves=15,
                            feature_fraction=0.6,
                            bagging_fraction=0.6,
                            lambda_l1=0.05,
                            objective='regression', 
                            verbose=-1)
    model_full.fit(X, y)
    return model_full

@task(name='Test model')
def test_model(X_test, y_test, y_train, model):
    """
    Tests the trained model against a test dataset and benchmarks against a naive mean predictor.
    
    Parameters:
        X_test (pd.DataFrame): Test feature matrix.
        y_test (pd.Series): True target values for testing.
        y_train (pd.Series): Target values from the training set, used for naive benchmarking.
        model (Model): The trained model to evaluate.
    """
    y_pred_test = model.predict(X_test)

    # Convert log-transformed values back to original scale
    original_y_test = np.exp(y_test) - 1
    original_y_preds = np.exp(y_pred_test) - 1

    # Calculate metrics
    mse_test = mean_squared_error(original_y_test, original_y_preds)
    mae_test = mean_absolute_error(original_y_test, original_y_preds)
    r2_test = r2_score(original_y_test, original_y_preds)

    # Benchmark metrics against a naive predictor that always predicts the mean of the training set
    mse_benchmark = mean_squared_error(original_y_test, np.full(len(original_y_test), np.mean(np.exp(y_train) - 1)))
    mae_benchmark = mean_absolute_error(original_y_test, np.full(len(original_y_test), np.mean(np.exp(y_train) - 1)))
    r2_benchmark = r2_score(original_y_test, np.full(len(original_y_test), np.mean(np.exp(y_train) - 1)))

    print("test_metrics: " + str(mse_test), str(mae_test), str(r2_test))
    print("benchmark_metrics: " + str(mse_benchmark), str(mae_benchmark), str(r2_benchmark))

@task(name="Convert To ONNX")
def convert_to_onnx(model, sample_input, onnx_file_path):
    """
    Converts a trained model to ONNX format and saves it to a specified file path.
    
    Parameters:
        model (Model): The trained model to convert.
        sample_input (np.array): A sample input for model inference, used for ONNX conversion.
        onnx_file_path (str): Path where the ONNX model will be saved.
    """
    onnx_gbt = convert(model, 'onnx', sample_input)
    try:
        # Attempt to make a prediction with the converted model to verify successful conversion
        out = onnx_gbt.predict(sample_input)
    except:
        print(f"Error converting to onnx")
    onnx.save_model(onnx_gbt.model, onnx_file_path)
    print(f"Model has been converted to ONNX and saved to {onnx_file_path}")

@action(name='Execution', log_prints=True)
def execution():
    """
    The main execution function that orchestrates the loading, processing, training, testing,
    and conversion of the model to ONNX format.
    """
    df = loading_and_processing()
    X_train, X_test, y_train, y_test = prepare_dataset(df)
    X_test[int(len(X_test) * 0.6):].to_csv("./example_token_vol.csv", header=False)
    model = train_model(X_train, y_train)
    test_model(X_test, y_test, y_train, model)
    
    onnx_file_path = "lgbm-token-vol.onnx"
    convert_to_onnx(model, X_test[:1].to_numpy(), onnx_file_path)

if __name__ == "__main__":
    action_deploy = Action(entrypoint=execution, name="lgbm-token-vol-action")
    action_deploy.serve(name="lgbm-token-vol-deployment")
