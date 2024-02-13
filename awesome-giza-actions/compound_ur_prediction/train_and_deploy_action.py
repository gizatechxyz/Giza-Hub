from giza_datasets import DatasetsHub, DatasetsLoader
from giza_actions.action import Action, action
from giza_actions.task import task
import polars as pl
import os
import certifi
import numpy as np
import gcsfs
import torch
import torch.nn as nn
import torch.optim as optim


os.environ["SSL_CERT_FILE"] = certifi.where()

loader = DatasetsLoader()
assets_to_keep = [
    "LINK",
    "SUSHI",
    "TUSD",
    "ZRX",
    "WBTC",
    "UNI",
    "COMP",
    "BAT",
    "YFI",
    "DAI",
    "USDT",
    "USDC",
    "ETH",
    "AAVE",
]


# Define the neural network architecture
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # First hidden layer
        self.fc2 = nn.Linear(64, 32)  # Third hidden layer
        self.fc3 = nn.Linear(32, 1)  # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def weighted_mean(apy: pl.Expr, tvlUsd: pl.Expr) -> pl.Expr:
    return (apy * tvlUsd).sum() / tvlUsd.sum()


def back_forward_fill(df, columns_to_fill):
    for col in columns_to_fill:
        # Step 1: Forward Fill - Fill nulls with the last observed non-null value
        df = df.with_columns(df[col].fill_null(strategy="forward").alias(col))
        # Step 2: Backward Fill - Fill any remaining nulls with the next observed non-null value
        df = df.with_columns(df[col].fill_null(strategy="backward").alias(col))

    return df


def check_for_missing_days(df):
    min_date = df.select(pl.min("date")).to_numpy()[0, 0]
    max_date = df.select(pl.max("date")).to_numpy()[0, 0]
    difference = (max_date - min_date).astype(int)
    assert len(df) >= difference


@task(name="Download and parse compound data")
def parse_compound_df(assets_to_keep):
    compound_df = loader.load("compound-daily-interest-rates")
    compound_df = compound_df.with_columns(
        [compound_df["symbol"].map_elements(lambda value: value[1:]).alias("symbol")]
    )

    # create the date column
    compound_df = compound_df.with_columns(
        (pl.col("timestamp") * 1000000)
        .cast(pl.Datetime)
        .dt.strftime("%Y-%m-%d")
        .alias("date")
    )

    min_comp_dt = compound_df.select(pl.min("date")).to_numpy()[0, 0]

    # prepare the compound table before merging
    compound_df = compound_df.drop("underlyingPriceUSD")
    compound_df = compound_df.drop("exchangeRate")
    compound_df = compound_df.drop("timestamp")
    compound_df = compound_df.drop("block_number")
    compound_df = compound_df.drop("totalSupply")
    compound_df = compound_df.drop("totalSupplyUSD")
    compound_df = compound_df.drop("totalBorrowUSD")
    compound_df = compound_df.group_by(["date", "symbol"]).agg(
        [
            pl.col("totalBorrows").sum().alias("totalBorrows"),
            pl.col("totalSupplyUnderlying").sum().alias("totalSupply"),
            # Weighted mean for borrowRate
            (
                (pl.col("borrowRate") * pl.col("totalBorrows")).sum()
                / pl.col("totalBorrows").sum()
            ).alias("borrowRate"),
            # Weighted mean for supplyRate
            (
                (pl.col("supplyRate") * pl.col("totalSupplyUnderlying")).sum()
                / pl.col("totalSupplyUnderlying").sum()
            ).alias("supplyRate"),
        ]
    )

    compound_df = compound_df.with_columns(
        (pl.col("totalBorrows") / pl.col("totalSupply")).alias("utilization_rate")
    )

    # Define the metrics you want to pivot on
    metrics = [
        "totalBorrows",
        "borrowRate",
        "supplyRate",
        "totalSupply",
        "utilization_rate",
    ]

    # Initialize a DataFrame to store the pivoted result for the first metric
    comp_grouped_df = compound_df.select(["date", "symbol"]).unique()

    # Loop through each metric, pivot the DataFrame, rename columns, and join the results
    for metric in metrics:
        # Pivot the DataFrame
        pivoted_df = compound_df.pivot(index="date", columns="symbol", values=metric)

        # Rename columns to include the metric as a suffix, except for the 'date' column
        renamed_columns = {
            col: f"{col}_{metric}" for col in pivoted_df.columns if col != "date"
        }
        pivoted_df = pivoted_df.rename(renamed_columns)

        # Join the pivoted DataFrame with the final DataFrame on 'date'
        comp_grouped_df = comp_grouped_df.join(pivoted_df, on="date", how="outer")
        comp_grouped_df = comp_grouped_df.drop("date_right")

    comp_grouped_df = comp_grouped_df.drop("symbol")
    comp_grouped_df = comp_grouped_df.sort("date")
    comp_grouped_df = comp_grouped_df.unique()
    comp_grouped_df = comp_grouped_df.with_columns(
        (pl.col("date").str.strptime(pl.Date, "%Y-%m-%d").alias("date"))
    )
    return comp_grouped_df, min_comp_dt


@task(name="Download and parse APY data")
def parse_apy_df(assets_to_keep):
    apy_df = loader.load("top-pools-apy-per-protocol")
    apy_df = apy_df.filter(apy_df["underlying_token"].is_in(assets_to_keep))
    apy_df = apy_df.rename({"underlying_token": "symbol"})
    apy_filter_protocols = ["yearn-finance", "spark", "aave-v3", "aave-v2", "thorchain"]
    apy_df = apy_df.filter(apy_df["project"].is_in(apy_filter_protocols))
    apy_df = apy_df.filter(apy_df["chain"].is_in(["Ethereum"]))
    apy_df = apy_df.drop("chain")
    # Group by 'date' and 'symbol', then calculate the weighted mean for 'apy'
    apy_df = apy_df.groupby(["date", "symbol"]).agg(
        weighted_mean(pl.col("apy"), pl.col("tvlUsd")).alias("mean_apy")
    )
    apy_df = apy_df.pivot(index="date", columns="symbol", values="mean_apy")
    apy_df = apy_df.sort("date")
    # check if we don't have any missing days
    check_for_missing_days(apy_df)

    apy_df = apy_df.drop("ETH")

    apy_df = back_forward_fill(apy_df, ["YFI", "DAI"])
    return apy_df


@task(name="Download and parse TVL data")
def parse_tvl_df(assets_to_keep, min_dt):
    tvl_df = loader.load("tvl-per-project-tokens")
    tvl_df = tvl_df.melt(id_vars=["date", "project"])
    tvl_df = tvl_df.rename({"variable": "symbol", "value": "tvl"})
    # nulls
    tvl_df = tvl_df.drop_nulls()

    # finally, filter for compound assets
    tvl_df = tvl_df.filter(tvl_df["symbol"].is_in(assets_to_keep))
    tvl_filter_protocols = [
        "yearn-finance",
        "aave-v2",
        "sushiswap",
        "aave-v3",
        "uniswap-v3",
        "curve-dex",
        "uniswap-v2",
        "balancer-v2",
    ]
    tvl_df = tvl_df.filter(tvl_df["project"].is_in(tvl_filter_protocols))
    tvl_df = tvl_df.group_by(["date", "symbol"]).agg(
        pl.col("tvl").sum().alias("total_tvl")
    )
    tvl_df = tvl_df.pivot(index="date", columns="symbol", values="total_tvl")
    tvl_df = tvl_df.sort("date")
    # check if we don't have any missing days
    check_for_missing_days(tvl_df)

    first_non_null_index = (
        tvl_df.with_row_count()
        .filter(pl.col("AAVE").is_not_null())["row_nr"]
        .gather([0])
    )[0]

    tvl_df = tvl_df.slice(first_non_null_index, tvl_df.height - first_non_null_index)
    tvl_df = tvl_df.drop("ETH")
    cols_to_fill = [i for i in tvl_df.columns if i != "date"]
    tvl_df = back_forward_fill(tvl_df, cols_to_fill)
    # Convert your string date to a Polars date for comparison
    min_comp_dt_pl = pl.lit(min_dt).str.strptime(pl.Date, "%Y-%m-%d")

    # Filter for rows where 'date' is greater than 'my_date'
    tvl_df = tvl_df.filter(tvl_df["date"] >= min_comp_dt_pl)

    for token in tvl_df.columns:
        if token != "date":
            tvl_df = tvl_df.rename({token: token + "_tvl"})
    return tvl_df


@task(name="Download and parse market cap data")
def parse_mcap_df(assets_to_keep, min_dt):
    fs = gcsfs.GCSFileSystem(verify=False)
    mcap_path = "gs://datasets-giza/tokens_daily_prices_mcap_volume/tokens_daily_prices_mcap_volume.parquet"
    with fs.open(mcap_path) as f:
        mcap_df = pl.read_parquet(f, use_pyarrow=True)

    mcap_df = mcap_df.filter(mcap_df["token"].is_in(assets_to_keep))
    mcap_df = mcap_df.rename({"token": "symbol"})
    price_df = mcap_df.select(["date", "price", "symbol"])
    price_df = price_df.pivot(index="date", columns="symbol", values="price")
    price_df = price_df.sort("date")
    price_df = price_df.filter(price_df["date"] >= min_dt)
    # check if we don't have any missing days
    check_for_missing_days(price_df)

    vol_df = mcap_df.select(["date", "volumes_last_24h", "symbol"])
    vol_df = vol_df.pivot(index="date", columns="symbol", values="volumes_last_24h")
    vol_df = vol_df.sort("date")
    vol_df = vol_df.filter(vol_df["date"] >= min_dt)
    # check if we don't have any missing days
    check_for_missing_days(vol_df)

    mcap_df = mcap_df.select(["date", "market_cap", "symbol"])
    mcap_df = mcap_df.pivot(index="date", columns="symbol", values="market_cap")
    mcap_df = mcap_df.sort("date")
    mcap_df = mcap_df.filter(mcap_df["date"] >= min_dt)
    # check if we don't have any missing days
    check_for_missing_days(mcap_df)

    price_df = back_forward_fill(price_df, [i for i in price_df.columns if i != "date"])
    vol_df = back_forward_fill(vol_df, [i for i in vol_df.columns if i != "date"])
    mcap_df = back_forward_fill(mcap_df, [i for i in mcap_df.columns if i != "date"])

    prices_tokens = list(price_df.columns)
    prices_tokens.remove("date")

    for token in prices_tokens:
        # Calculate log returns
        log_returns = pl.col(token) / pl.col(token).shift(1)
        log_returns = log_returns.map_elements(
            lambda x: np.log(x) if x is not None else None
        ).alias(token + "_logreturn")

        # Add the log returns column to the DataFrame
        price_df = price_df.with_columns(log_returns)

    window_size = 7  # 1 week rolling std

    for token in prices_tokens:
        vol = price_df.select(
            pl.col(token + "_logreturn")
            .rolling_std(window_size)
            .alias(token + "_volatility")
        )
        price_df = price_df.with_columns(vol)
        price_df = price_df.drop(token)

    for token in vol_df.columns:
        if token != "date":
            vol_df = vol_df.rename({token: token + "_volume"})

    for token in mcap_df.columns:
        if token != "date":
            mcap_df = mcap_df.rename({token: token + "_mcap"})

    return price_df, vol_df, mcap_df


@task(name="Combine dataframes")
def combine_dfs(comp_grouped_df, apy_df, tvl_df, vol_df, mcap_df, price_df):
    final_df = comp_grouped_df.join(apy_df, on="date", how="outer")
    final_df = final_df.drop("date_right")
    final_df = final_df.join(tvl_df, on="date", how="outer")
    final_df = final_df.drop("date_right")
    final_df = final_df.join(vol_df, on="date", how="outer")
    final_df = final_df.drop("date_right")
    final_df = final_df.join(mcap_df, on="date", how="outer")
    final_df = final_df.drop("date_right")
    final_df = final_df.join(price_df, on="date", how="outer")
    final_df = final_df.drop("date_right")
    final_df = final_df.filter(pl.col("date").is_not_null())
    return final_df


@task(name="Clean the data")
def clean_final(final_df):
    null_checks = [pl.col(col).is_null().cast(pl.UInt32) for col in final_df.columns]

    # Sum the results of these expressions to count the number of nulls in each row
    null_count_expr = sum(null_checks).alias("null_count")

    # Add the null count as a new column to the original DataFrame
    final_df = final_df.with_columns(null_count_expr)
    final_df = final_df.filter(pl.col("null_count") < 60)
    final_df = final_df.drop("null_count")
    # Create an expression to count nulls for each column
    null_counts = [
        pl.col(column).is_null().sum().alias(column)
        for column in final_df.columns
        if column != "date"
    ]

    # Apply the expressions to the DataFrame and collect the result
    null_count_df = final_df.select(null_counts)

    df_long = null_count_df.melt(
        id_vars=[], value_vars=null_count_df.columns, value_name="Values"
    )
    df_long = df_long.filter(pl.col("Values") > 600)
    cols_to_remove = df_long["variable"].to_list()
    final_df = final_df.drop(cols_to_remove)
    final_df = back_forward_fill(final_df)
    final_df = final_df.with_columns(
        pl.col("USDC_utilization_rate").shift(-1).alias("USDC_utilization_rate")
    )

    final_df = final_df.drop_nulls()

    final_df = final_df.sort("date")
    return final_df


@task(name="Get train and test sets")
def get_train_test(final_df):

    X_df = final_df.drop(["USDC_utilization_rate", "date"])
    Y_df = final_df.select(["USDC_utilization_rate"])

    X_pandas = X_df.to_pandas()
    Y_pandas = Y_df.to_pandas()

    # Split the data into training and testing sets based on the time order
    # Assuming 80% for training and 20% for testing as an example
    split_index = int(len(X_pandas) * 0.8)
    X_train, X_test = X_pandas[:split_index], X_pandas[split_index:]
    y_train, y_test = Y_pandas[:split_index], Y_pandas[split_index:]
    return X_train, X_test, y_train, y_test


@task(name="Train the model")
def train_model(model, X_train, y_train, X_test, y_test, epochs=100000, lr=0.001):
    X_train_tensor = torch.tensor(X_train.to_numpy().astype(np.float32))
    y_train_tensor = torch.tensor(y_train.to_numpy().astype(np.float32).reshape(-1, 1))
    X_test_tensor = torch.tensor(X_test.to_numpy().astype(np.float32))
    y_test_tensor = torch.tensor(y_test.to_numpy().astype(np.float32).reshape(-1, 1))

    # Instantiate the model
    input_size = X_train.shape[1]
    model = SimpleNN(input_size)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 10000 == 0:  # Print loss every 10 epochs
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
        test_loss = criterion(y_pred_tensor, y_test_tensor)
        test_rmse = torch.sqrt(test_loss)

    print(f"Model RMSE: {test_rmse.item()}")


@task(name="Export model to ONNX")
def export_to_onnx(model, X_train, onnx_model_path):
    sample_input = torch.randn(1, X_train.shape[1], dtype=torch.float32)

    # Export the model
    torch.onnx.export(
        model,  # Model being exported
        sample_input,  # Model input (or a tuple for multiple inputs)
        onnx_model_path,  # Where to save the model
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

    print(f"Model has been converted to ONNX and saved to {onnx_model_path}")


@task(name="Load and process data")
def load_and_process():
    comp_df, min_dt = parse_compound_df(assets_to_keep)
    apy_df = parse_apy_df(assets_to_keep)
    tvl_df = parse_tvl_df(assets_to_keep, min_dt)
    price_df, vol_df, mcap_df = parse_mcap_df(assets_to_keep, min_dt)
    final_df = combine_dfs(comp_df, apy_df, tvl_df, vol_df, mcap_df, price_df)
    final_df = clean_final(final_df)
    return final_df


@action(name=f"Execution", log_prints=True)
def execution():
    df = load_and_process()
    X_train, X_test, y_train, y_test = get_train_test(df)
    model = SimpleNN()
    train_model(model, X_train, y_train, X_test, y_test)
    export_to_onnx(model, X_train, "ff_nn_compound_ur_prediction.onnx")


if __name__ == "__main__":
    action_deploy = Action(entrypoint=execution, name="compound_ur_prediction")
    action_deploy.serve(name="compound_ur_prediction")
