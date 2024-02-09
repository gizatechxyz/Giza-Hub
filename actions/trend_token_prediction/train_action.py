import polars as pl
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from giza_actions.task import task

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import onnxruntime as rt
from giza_actions.action import Action, action
from giza_actions.task import task
from giza_actions.model import GizaModel
import numpy as np
import torch.nn.functional as F

import certifi
import os
os.environ['SSL_CERT_FILE'] = certifi.where()
from giza_datasets import DatasetsLoader

TARGET_LAG = 1 # target lagg
TOKEN_NAME = "WETH" #name of the token that we wants to predict 
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
#@task(name=f'Training loop')
def train_model(model, criterion, optimizer, X_train, y_train, epochs=100):
    model.train()  # Asegura que el modelo esté en modo entrenamiento
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

#@task(name=f'Generate predictions')
def predict(model, X_test):
    model.eval()  # Establece el modelo en modo evaluación
    X_test_tensor = torch.tensor(X_test.astype(np.float32))
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
    return y_pred_tensor.numpy()


# Task Definitions
def resize_images(images):
    return np.array([zoom(image[0], (0.5, 0.5)) for image in images])

#@task(name=f'prepare train and test data')
def prepare_train_test(df_train, df_test):
    for col in df_train.columns:
        mean_val = df_train[col].mean()
        std_dev = df_train[col].std() if df_train[col].std() != 0 else 1
        df_train = df_train.with_columns(((df_train[col].fill_null(mean_val) - mean_val) / std_dev).alias(col))
        df_test = df_test.with_columns(((df_test[col].fill_null(mean_val) - mean_val) / std_dev).alias(col))
    return df_train, df_test

#@task(name=f'Null treatment')
def delete_null_columns(df, null_percentaje):
    threshold = df.shape[0] * null_percentaje
    columns_to_keep = [
        col_name for col_name in df.columns if df[col_name].null_count() <= threshold
    ]
    return df.select(columns_to_keep)

#@task(name=f'Calculate and print metrics')
def print_classification_metrics(y_test, y_pred, y_pred_proba = None):
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
#@task(name=f'calculate lag correlations')
def calculate_lag_correlations(df, lags=[1, 3, 7, 15]):
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

#@task(name=f'Main dateset manipulation')
def main_dateset_manipulation():
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
    # Incorporamos la lógica para día de la semana y mes del año
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
#@task(name=f'APY dateset manipulation')
def apy_dateset_manipulation():
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
#@task(name=f'TVL dateset manipulation')
def tvl_dateset_manipulation():
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

#@task(name=f'Prepare and train')
def prepare_and_train(X_train,y_train):
    X_train_np = X_train.to_numpy().astype(np.float32)
    y_train_np = y_train.to_numpy().astype(np.float32).reshape(-1, 1)


    input_size = X_train_np.shape[1]
    model = SimpleNN(input_size)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    model = train_model(model, criterion, optimizer, X_train_np, y_train_np, epochs=100)

@task(name=f'Prepare datasets')
def prepare_datasets(df):
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
    X_test_np = X_test.to_numpy().astype(np.float32)
    y_pred = predict(model, X_test_np)
    y_pred_labels = (y_pred >= 0.5).astype(int)
    print(y_pred)
    print(y_pred_labels)
    print(y_test)
    print_classification_metrics(y_test, y_pred_labels, y_pred)
    
@task(name="Convert To ONNX")
def convert_to_onnx(model, sample_len, onnx_file_path):
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
    df = load_and_df_processing()
    X_train, X_test, y_train, y_test = prepare_datasets(df)
    model = prepare_and_train(X_train,y_train)
    test_model(X_test, y_test, model)

    # Convert to ONNX
    onnx_file_path = "pytorch-token-trend_action_model.onnx"
    convert_to_onnx(model, X_test.shape[1], onnx_file_path)

if __name__ == "__main__":
    action_deploy = Action(entrypoint=execution, name="pytorch-token-trend-action")
    action_deploy.serve(name="pytorch-token-trend-deployment")
