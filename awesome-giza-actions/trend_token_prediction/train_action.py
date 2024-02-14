import polars as pl
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import onnxruntime as rt

# GIZA stack
from giza_actions.task import task
from giza_datasets import DatasetsLoader
from giza_actions.action import Action, action
from giza_actions.task import task
from giza_actions.model import GizaModel

import certifi
import os
os.environ['SSL_CERT_FILE'] = certifi.where()


TARGET_LAG = 1 # target lagg
TOKEN_NAME = "WETH" # Choose one of the available tokens in the main dataset.
STARTER_DATE = pl.datetime(2022, 6, 1)
LOADER = DatasetsLoader()

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.init_weights()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

def train_model(model, criterion, optimizer, X_train, y_train, epochs=100):
    """
    Trains a neural network model using the specified training data, loss criterion, and optimizer.

    Parameters:
    - model: The neural network model to be trained.
    - criterion: The loss function used to evaluate the model's performance.
    - optimizer: The optimization algorithm used to update the model's weights.
    - X_train: Training data features as a NumPy array.
    - y_train: Training data labels/targets as a NumPy array.
    - epochs (optional): The number of training epochs (default is 100).

    Returns:
    The trained model.
    """
    model.train()
    X_train_tensor = torch.tensor(X_train.astype(np.float32))
    y_train_tensor = torch.tensor(y_train.astype(np.float32).reshape(-1, 1))
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")
    return model

def predict(model, X_test):
    """
    Makes predictions using a trained model on the provided test data.

    Parameters:
    - model: The trained neural network model.
    - X_test: Test data features as a NumPy array.

    Returns:
    Predictions as a NumPy array.
    """
    model.eval()
    X_test_tensor = torch.tensor(X_test.astype(np.float32))
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
    return y_pred_tensor.numpy()

def prepare_train_test(df_train, df_test):
    """
    Preprocesses training and test dataframes by standardizing columns based on training data statistics.

    Parameters:
    - df_train: The training dataframe.
    - df_test: The test dataframe.

    Returns:
    A tuple containing the preprocessed training and test dataframes.
    """
    for col in df_train.columns:
        mean_val = df_train[col].mean()
        std_dev = df_train[col].std() if df_train[col].std() != 0 else 1
        df_train = df_train.with_columns(((df_train[col].fill_null(mean_val) - mean_val) / std_dev).alias(col))
        df_test = df_test.with_columns(((df_test[col].fill_null(mean_val) - mean_val) / std_dev).alias(col))
    return df_train, df_test

def delete_null_columns(df, null_percentaje):
    """    
    Removes columns from a dataframe where the percentage of null values exceeds a specified threshold.

    Parameters:
    - df: The dataframe to process.
    - null_percentage: The threshold percentage of null values for column removal.

    Returns:
    The dataframe with columns removed based on the null value threshold.
    """
    threshold = df.shape[0] * null_percentaje
    columns_to_keep = [
        col_name for col_name in df.columns if df[col_name].null_count() <= threshold
    ]
    return df.select(columns_to_keep)

def print_classification_metrics(y_test, y_pred, y_pred_proba = None):
    """
    Prints classification metrics including accuracy, precision, recall, F1 score, and optionally AUC.

    Parameters:
    - y_test: The true labels for the test data.
    - y_pred: The predicted labels for the test data.
    - y_pred_proba (optional): The predicted probabilities for the test data. If provided, AUC will be calculated and printed.

    This function computes and prints the confusion matrix, accuracy, precision, recall, and F1 score of the classification model's predictions. If predicted probabilities are provided, it also computes and prints the AUC score.
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    if y_pred_proba is not None:
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"AUC: {auc}")

def calculate_lag_correlations(df, lags=[1, 3, 7, 15]):
    """
    Calculates and returns the lagged correlations between different tokens' prices in the dataset.

    Parameters:
    - df: The input dataframe containing at least 'token', 'date', and 'price' columns.
    - lags (optional): A list of integers specifying the lag days for which to calculate correlations.

    The function iterates over each unique token pair in the dataset, computes the correlation of their prices at
    specified lags, and returns a dictionary with these correlations. The dictionary keys are formatted as
    "{base_token}_vs_{compare_token}" with sub-keys for each lag indicating the correlation at that lag.

    Returns:
    A dictionary of lagged price correlations for each token pair in the dataset.
    """
    correlations = {}
    tokens = df.select("token").unique().to_numpy().flatten()
    for base_token in tokens:
        for compare_token in tokens:
            if base_token == compare_token:
                continue
            base_df = df.filter(pl.col("token") == base_token).select(["date", "price"]).sort("date")
            compare_df = df.filter(pl.col("token") == compare_token).select(["date", "price"]).sort("date")
            merged_df = base_df.join(compare_df, on="date", suffix="_compare")
            key = f"{base_token}_vs_{compare_token}"
            correlations[key] = {}
            for lag in lags:
                merged_df_lagged = merged_df.with_columns(pl.col("price_compare").shift(lag))
                corr_df = merged_df_lagged.select(
                    pl.corr("price", "price_compare").alias("correlation")
                )
                corr = corr_df.get_column("correlation")[0]
                correlations[key][f"lag_{lag}_days"] = corr
                
    return correlations

def main_dateset_manipulation():
    """
    Performs the main dataset manipulation including loading the dataset, generating features, and calculating correlations.

    This function executes several steps:
    - Loads the 'tokens-daily-prices-mcap-volume' dataset.
    - Filters the dataset for a specified token.
    - Calculates various features such as price differences and trends over different days.
    - Adds day of the week, month of the year, and year as features.
    - Calculates lag correlations between the specified token and all other tokens in the dataset.
    - Identifies the top 10 correlated tokens based on 15-day lag correlation.
    - Joins features of the top 10 correlated tokens to the main dataframe.
    
    Returns:
    A DataFrame containing the original features along with the newly calculated features and correlations for further analysis.
    """
    daily_token_prices = LOADER.load('tokens-daily-prices-mcap-volume')

    df_main = daily_token_prices.filter(pl.col("token") == TOKEN_NAME)

    df_main = df_main.with_columns(
        ((pl.col("price").shift(-TARGET_LAG) - pl.col("price")) > 0).cast(pl.Int8).alias("target")
    )

    df_main = df_main.with_columns([
        (pl.col("price").diff().alias("diff_price_1_days_ago")),
        ((pl.col("price") - pl.col("price").shift(1)) > 0).cast(pl.Int8).alias("trend_1_day"),
        (pl.col("price").diff(n = 3).alias("diff_price_3_days_ago")),
        ((pl.col("price") - pl.col("price").shift(3)) > 0).cast(pl.Int8).alias("trend_3_day"),
        (pl.col("price").diff(n = 7).alias("diff_price_7_days_ago")),
        ((pl.col("price") - pl.col("price").shift(7)) > 0).cast(pl.Int8).alias("trend_7_day"),
        (pl.col("price").diff(n = 15).alias("diff_price_15_days_ago")),
        ((pl.col("price") - pl.col("price").shift(15)) > 0).cast(pl.Int8).alias("trend_15_day"),
        (pl.col("price").diff(n = 30).alias("diff_price_30_days_ago")),
        ((pl.col("price") - pl.col("price").shift(30)) > 0).cast(pl.Int8).alias("trend_30_day")
        ]
    )
    df_main = df_main.with_columns([
        pl.col("date").dt.weekday().alias("day_of_week"),
        pl.col("date").dt.month().alias("month_of_year"),
        pl.col("date").dt.year().alias("year")
    ])

    correlations = calculate_lag_correlations(daily_token_prices)

    data = []
    for tokens, lags in correlations.items():
        base_token, compare_token = tokens.split('_vs_')
        for lag, corr_value in lags.items():
            data.append({'Base Token': base_token, 'Compare Token': compare_token, 'Lag': lag, 'Correlation': corr_value})

    df_correlations = pl.DataFrame(data)

    top_10_correlated_coins =df_correlations.filter((pl.col("Base Token") == TOKEN_NAME) & 
                                                    (pl.col("Lag") == "lag_15_days")).sort(by="Correlation", descending = True)["Compare Token"].to_list()[0:10]

    for token in top_10_correlated_coins:
        df_token = daily_token_prices.filter(pl.col("token") == token)
        df_token_features = df_token.with_columns([
            pl.col("price").diff(n = 1).alias(f"diff_price_1_days_ago{token}"),
            ((pl.col("price") - pl.col("price").shift(1)) > 0).cast(pl.Int8).alias(f"trend_1_day{token}"),
            pl.col("price").diff(n = 3).alias(f"diff_price_3_days_ago{token}"),
            ((pl.col("price") - pl.col("price").shift(3)) > 0).cast(pl.Int8).alias(f"trend_3_day{token}"),
            pl.col("price").diff(n = 7).alias(f"diff_price_7_days_ago{token}"),
            ((pl.col("price") - pl.col("price").shift(7)) > 0).cast(pl.Int8).alias(f"trend_7_day{token}"),
            pl.col("price").diff(n = 15).alias(f"diff_price_15_days_ago{token}"),
            ((pl.col("price") - pl.col("price").shift(15)) > 0).cast(pl.Int8).alias(f"trend_15_day{token}"),
        ]).select([
            pl.col("date"),
            f"diff_price_1_days_ago{token}",
            f"diff_price_3_days_ago{token}",
            f"diff_price_7_days_ago{token}",
            f"diff_price_15_days_ago{token}",
            f"trend_1_day{token}",
            f"trend_3_day{token}",
            f"trend_7_day{token}",
            f"trend_15_day{token}",
        ])
        df_main = df_main.join(df_token_features, on="date", how="left")
        return df_main
    
def apy_dateset_manipulation():
    """
    Manipulates the APY dataset for a specific token to prepare it for analysis.

    Returns:
    A pivoted DataFrame focused on the specified token, with each row representing a date and each column representing a different project's TVL and APY.
    """
    apy_df = LOADER.load("top-pools-apy-per-protocol")

    apy_df = apy_df.filter(pl.col("underlying_token").str.contains(TOKEN_NAME))

    apy_df = apy_df.with_columns(
        pl.col("project") + "_" + pl.col("chain") +  pl.col("underlying_token")
    )
    apy_df = apy_df.drop(["underlying_token", "chain"])

    unique_projects = apy_df.filter(pl.col("date") <= STARTER_DATE).select("project").unique()

    apy_df_token = apy_df.join(
        unique_projects, 
        on="project", 
        how="inner"
    )

    apy_df_token = apy_df_token.pivot(
        index="date",
        columns="project",
        values=["tvlUsd", "apy"]
    )
    return apy_df_token

def tvl_dateset_manipulation():
    """
    Manipulates the TVL (Total Value Locked) dataset for a specific token to prepare it for analysis.

    Returns:
    A pivoted DataFrame focused on the specified token's TVL, with each row representing a date and each column representing a different project's TVL.
    """
    tvl_df = LOADER.load("tvl-per-project-tokens")

    tvl_df = tvl_df.filter(tvl_df[["date", "project"]].is_duplicated() == False)
    tvl_df = tvl_df.filter(tvl_df["date"] > STARTER_DATE)

    tvl_df = tvl_df[[TOKEN_NAME, "project", "date"]].pivot(
        index="date",
        columns="project",
        values= TOKEN_NAME
    )
    return tvl_df

@task(name=f'Join and postprocessing')
def load_and_df_processing():
    """
    Loads and processes the main, APY, and TVL datasets, joining them on the date column and performing postprocessing.

    Returns:
    A DataFrame ready for further analysis or model training, containing combined and processed features from all datasets.
    """
     
    df_main = main_dateset_manipulation()
    apy_df = apy_dateset_manipulation()
    tvl_df = tvl_dateset_manipulation()

    df_main = df_main.join(tvl_df, on = "date", how = "inner")
    df_main = df_main.join(apy_df, on = "date", how = "inner")

    num_rows_to_select = len(df_main) - TARGET_LAG
    df_main = df_main.slice(0, num_rows_to_select)

    # Some of the extra tokens we added do not have much historical information, so we raised the minimum date of our dataset a little bit.
    df_main = df_main.filter(pl.col("year") >= 2022)
    df_main = df_main.drop(["token","market_cap"])
    df_main = delete_null_columns(df_main, 0.2)
    return df_main

@task(name=f'prepare and train')
def prepare_and_train(X_train,y_train):
    """
    Prepares the training data and trains the neural network model.

    Parameters:
    - X_train: Feature DataFrame for training.
    - y_train: Target DataFrame for training.

    This function converts the training data to NumPy arrays, initializes the neural network model, and trains it using the specified features and targets.

    Returns:
    The trained neural network model.
    """
    X_train_np = X_train.to_numpy().astype(np.float32)
    y_train_np = y_train.to_numpy().astype(np.float32).reshape(-1, 1)


    input_size = X_train_np.shape[1]
    model = SimpleNN(input_size)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    model = train_model(model, criterion, optimizer, X_train_np, y_train_np, epochs=100)
    return model

@task(name=f'Prepare datasets')
def prepare_datasets(df):
    """
    Prepares the datasets for training and testing by selecting features, splitting the data, and preprocessing.

    Parameters:
    - df: The main DataFrame containing features and targets.

    Returns:
    Prepared feature and target DataFrames for both training and testing.
    """
    features = list(set(df.columns) - set(["date","month_of_year", "price", "target"]))
    cutoff_index = int(len(df) * 0.85)
    df_train = df[:cutoff_index]
    df_test = df[cutoff_index:]
    X_train, X_test = prepare_train_test(df_train[features], df_test[features])
    y_train = df_train.select('target')
    y_test = df_test.select('target')
    return X_train, X_test, y_train, y_test

@task(name=f'Test model')
def test_model(X_test, y_test, model):
    """
    Tests the trained model using the test dataset and prints classification metrics.

    Parameters:
    - X_test: Feature DataFrame for testing.
    - y_test: Actual target values for the test dataset.
    - model: The trained neural network model.
    """
    X_test_np = X_test.to_numpy().astype(np.float32)
    y_pred = predict(model, X_test_np)
    y_pred_labels = (y_pred >= 0.5).astype(int)
    print_classification_metrics(y_test, y_pred_labels, y_pred)
    
@task(name="Convert To ONNX")
def convert_to_onnx(model, sample_len, onnx_file_path):
    """
    Converts a PyTorch model to the ONNX format and saves it to a specified file path.

    Parameters:
    - model: The PyTorch model to be converted.
    - sample_len: The length of the input sample, specifying the input size.
    - onnx_file_path: The file path where the ONNX model will be saved.

    This function takes a trained PyTorch model and a sample input size, exports the model to the ONNX format,
    and saves it to the provided file path. It specifies model input/output names and handles dynamic batch sizes
    for flexibility in model deployment.
    """
    sample_input = torch.randn(1, sample_len, dtype=torch.float32)

    torch.onnx.export(
        model,  # Model being exported
        sample_input,  # Model input (or a tuple for multiple inputs)
        onnx_file_path,  # Where to save the model
        export_params=True,  # Store the trained parameter weights inside the model file
        opset_version=11,  # ONNX version to export the model to
        do_constant_folding=True,  # Whether to execute constant folding for optimization
        input_names=["input"],  # Model's input names
        output_names=["output"],  # Model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # Variable length axes
            "output": {0: "batch_size"},
        },
    )
    print(f"Model has been converted to ONNX and saved to {onnx_file_path}")
    
@action(name=f'Execution', log_prints=True )
def execution():
    """
    Main execution action that processes data, trains a model, tests the model, and converts it to ONNX format.

    This action performs the following steps:
    - Loads and processes the main dataset.
    - Prepares datasets for training and testing.
    - Saves a subset of the test dataset for example predictions.
    - Trains a neural network model using the prepared training dataset.
    - Tests the trained model using the test dataset and prints classification metrics.
    - Converts the trained model to the ONNX format for deployment.

    The ONNX model is saved to a predefined file path, and the action demonstrates an end-to-end workflow from data
    preprocessing to model deployment in the ONNX format.
    """
    df = load_and_df_processing()
    X_train, X_test, y_train, y_test = prepare_datasets(df)
    X_test[int(len(X_test) * 0.6):].write_csv("./example_token_trend.csv")
    model = prepare_and_train(X_train,y_train)
    test_model(X_test, y_test, model)
    
    onnx_file_path = "pytorch-token-trend_action_model.onnx"
    convert_to_onnx(model, X_test.shape[1], onnx_file_path)

if __name__ == "__main__":
    action_deploy = Action(entrypoint=execution, name="pytorch-token-trend-action")
    action_deploy.serve(name="pytorch-token-trend-deployment")
