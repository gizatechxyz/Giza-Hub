from giza_datasets import DatasetsLoader
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from lightgbm import LGBMRegressor
import shap
from financial_features import add_financial_features, calculate_future_volatility

TARGET_LAG = 7 # target lagg
TOKEN_NAME = "WETH" # Choose one of the available tokens in the main dataset.
STARTER_DATE = pl.datetime(2022, 3, 1)
LOADER = DatasetsLoader()

def calculate_lagged_correlations(df, target_token, lag_days=15, n=10):
    """
    Calculates the correlations between the lagged prices of various tokens and the price of a target token.
    
    Parameters:
    - df: DataFrame containing 'date', 'token', and 'price' columns.
    - target_token: The token whose price is to be compared against others.
    - lag_days: The number of days of lag to apply when calculating the correlation.
    - n: The maximum number of tokens with the highest correlation to return.
    
    Returns:
    - List of tokens sorted by their correlation with the target token, in descending order, limited to the top n tokens.
    """
    df.sort_values(by='date', inplace=True)
    pivoted_df = df.pivot(index='date', columns='token', values='price')
    lagged_df = pivoted_df.shift(periods=lag_days)
    target_series = pivoted_df[target_token]

    correlations = {}

    for token in lagged_df.columns:
        if token != target_token:  # Skip comparing the target token with itself
            valid_indices = target_series.notna() & lagged_df[token].notna()
            corr = target_series[valid_indices].corr(lagged_df[token][valid_indices])
            correlations[token] = corr

    sorted_tokens = sorted(correlations, key=correlations.get, reverse=True)[:n]
    
    return sorted_tokens

def remove_columns_with_nulls_above_threshold(df, threshold=0.5):
    """
    Removes columns from a DataFrame where the percentage of null values exceeds a specified threshold.
    
    Parameters:
    - df: Input DataFrame.
    - threshold: Threshold percentage of null values to drop the column.
    
    Returns:
    - DataFrame without columns exceeding the null value threshold.
    """
    null_percentage = df.isnull().mean()
    columns_to_drop = null_percentage[null_percentage > threshold].index
    df_filtered = df.drop(columns=columns_to_drop)
    
    return df_filtered

def dimensionality_reduction(X, y, corr_threshold = 0.85, n_features_RFE = 25):
    """
    Reduces the dimensionality of the feature space by removing highly correlated features and using Recursive Feature Elimination.
    
    Parameters:
    - X: DataFrame of features.
    - y: Series or array of target variable.
    - corr_threshold: Threshold for the correlation above which features should be removed.
    - n_features_RFE: Number of features to select with Recursive Feature Elimination.
    
    Returns:
    - X: DataFrame of features after dimensionality reduction.
    """
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    X = X.drop(columns=to_drop)

    # RFE
    estimator = LGBMRegressor()
    selector = RFE(estimator, n_features_to_select=n_features_RFE, step=3)
    selector = selector.fit(X, y)
    X = X.iloc[:, selector.support_]
    return X

def plot_shap(model, X):
    """
    Plots SHAP values for the features in the dataset to interpret the model's predictions.
    
    Parameters:
    - model: The trained model.
    - X: DataFrame of features used by the model.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X)

def daily_price_dateset_manipulation():
    """
    Manipulates and prepares the daily price dataset for modeling, including lagged correlation calculation and feature engineering.
    
    Returns:
    - df_final: The final DataFrame ready for modeling.
    """
    daily_token_prices = LOADER.load('tokens-daily-prices-mcap-volume')
    df = daily_token_prices.to_pandas()
    correlations = calculate_lagged_correlations(df, target_token=TOKEN_NAME)

    df_final = pd.DataFrame()

    for token in [TOKEN_NAME] + correlations:
        df_token = df[df['token'] == token].copy()
        df_features = add_financial_features(df_token)
        df_features.drop("token", axis = 1, inplace = True)
        if token == TOKEN_NAME:
            df_features['future_vol'] = calculate_future_volatility(df_features, TARGET_LAG)
            df_features = df_features.dropna(subset = ["future_vol"])
            df_features = df_features.add_prefix(f"{token}_")
            df_final = df_features
            continue
        df_features = df_features.add_prefix(f"{token}_")
        df_final = pd.merge(df_final, df_features, on = "date", how = "left")
    df_final.reset_index(inplace=True)
    return df_final
    
def apy_dateset_manipulation():
    """
    Manipulates the APY dataset to focus on specific tokens and reshape it for easier analysis.
    
    Returns:
    - apy_df_token: The manipulated APY DataFrame.
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
    Manipulates the TVL dataset to focus on specific projects and tokens, reshaping it for analysis.
    
    Returns:
    - tvl_df: The manipulated TVL DataFrame.
    """
    tvl_df = LOADER.load("tvl-per-project-tokens")
    tvl_df = tvl_df.filter(tvl_df[["date", "project"]].is_duplicated() == False)

    tvl_df = tvl_df[[TOKEN_NAME, "project", "date"]].pivot(
        index="date",
        columns="project",
        values= TOKEN_NAME
    )
    return tvl_df