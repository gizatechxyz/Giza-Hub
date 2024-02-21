import polars as pl
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import onnxruntime as rt
import pandas as pd
import torch.onnx as onnx

# GIZA stack
from giza_actions.task import task
from giza_datasets import DatasetsLoader
from giza_actions.action import Action, action
from giza_actions.task import task
from giza_actions.model import GizaModel

import certifi
import os
os.environ['SSL_CERT_FILE'] = certifi.where()

@task(name=f'Load and Merge Data')
def load_and_df_processing():
     
    loader = DatasetsLoader()

    df_liquidations_v2 = loader.load("aave-liquidationsV2").to_pandas()
    df_liquidations_v3 = loader.load("aave-liquidationsV3").to_pandas()
    df_liquidations = pd.concat([df_liquidations_v2, df_liquidations_v3], keys=['v2', 'v3'])
    df_liquidations['version'] = df_liquidations.index.get_level_values(0)
    df_liquidations = df_liquidations[df_liquidations['token_col'].isin(["WETH","USDC","LINK","WBTC","stETH"])]
    token_list = ["WETH","USDC","LINK","WBTC","stETH"]
    df_liquidations = df_liquidations[df_liquidations['token_debt'].isin(["USDC","USDT","DAI","WETH","WBTC"])]
    token_list += ["USDT","DAI"]
    df_liquidations = df_liquidations.drop(columns=['user', 'liquidator', 'version','col_contract_address','debt_contract_address'])
    df_liquidations_daily = df_liquidations.groupby(['day', 'token_col', 'token_debt']).agg({
    'collateral_amount': 'sum',
    'col_value_USD': 'sum',
    'col_current_value_USD': 'sum',
    'debt_amount': 'sum',
    'debt_value_USD': 'sum',
    'debt_current_value_USD': 'sum'}).reset_index()

    df_deposits_borrows_v2 = loader.load("aave-daily-deposits-borrowsv2").to_pandas()
    df_deposits_borrows_v3 = loader.load("aave-daily-deposits-borrowsv3").to_pandas()
    df_deposits_borrows = pd.concat([df_deposits_borrows_v2, df_deposits_borrows_v3], keys=['v2', 'v3'])
    df_deposits_borrows['version'] = df_deposits_borrows.index.get_level_values(0)
    df_deposits_borrows = df_deposits_borrows.groupby(['day', 'symbol']).sum().reset_index()
    df_deposits_borrows = df_deposits_borrows.drop(columns=['version','contract_address'])
    df_deposits_borrows = df_deposits_borrows[df_deposits_borrows['symbol'].isin(token_list)]

    df_daily_token = loader.load('tokens-daily-prices-mcap-volume').to_pandas()
    token_list_upper = [token.upper() for token in token_list]
    df_daily_token = df_daily_token[df_daily_token['token'].isin(token_list_upper)]

    df_liquidations_debts = df_liquidations_daily.drop(columns=['token_col','collateral_amount', 'col_value_USD', 'col_current_value_USD'])

    earliest_day = df_liquidations_debts['day'].min()
    filled_rows = pd.DataFrame()

    for unique_token in df_liquidations_debts['token_debt'].unique():
        token_col_df = df_liquidations_debts[df_liquidations_debts['token_debt'] == unique_token]
        
        missing_days = pd.date_range(start=earliest_day, end=token_col_df['day'].max(), freq='D').difference(token_col_df['day'])
        
        missing_rows = pd.DataFrame({
            'day': missing_days,
            'token_debt': unique_token,
            'debt_amount': 0,
            'debt_value_USD': 0,
            'debt_current_value_USD': 0
        })
        
        filled_rows = pd.concat([missing_rows,filled_rows])

    df_liquidations_debts_filled = pd.concat([df_liquidations_debts, filled_rows])
    df_liquidations_debts_filled = df_liquidations_debts_filled.sort_values('day')
    df_liquidations_debts_filled = df_liquidations_debts_filled.reset_index(drop=True)

    earliest_day_minus_7 = earliest_day - pd.DateOffset(days=7)
    df_deposits_borrows_filtered = df_deposits_borrows[df_deposits_borrows['day'] >= earliest_day_minus_7]
    df_deposits_borrows_filtered = df_deposits_borrows_filtered.rename(columns={'symbol': 'token_debt',})

    merged_df = pd.merge(df_liquidations_debts_filled, df_deposits_borrows_filtered, how='outer', on=['day', 'token_debt'])
    merged_df.fillna(0, inplace=True)

    df_daily_token_filtered = df_daily_token[df_daily_token['date'] >= earliest_day_minus_7]
    df_daily_token_filtered = df_daily_token_filtered.rename(columns={'token': 'token_debt','date':'day'})
    merged_df['token_debt'] = merged_df['token_debt'].str.upper()
    merged_df = pd.merge(merged_df, df_daily_token_filtered, how='outer', on=['day', 'token_debt'])
    merged_df.fillna(0, inplace=True)

    return merged_df, earliest_day


@task(name=f'Feature Engineering')
def feature_engineering(merged_df, earliest_day):
    merged_df['deposits_volume_avg_3d'] = merged_df['deposits_volume'].rolling(window=3, min_periods=1).mean()
    merged_df['borrows_volume_avg_3d'] = merged_df['borrows_volume'].rolling(window=3, min_periods=1).mean()
    merged_df['market_cap_avg_3d'] = merged_df['market_cap'].rolling(window=3, min_periods=1).mean()
    merged_df['volumes_last_24h_avg_3d'] = merged_df['volumes_last_24h'].rolling(window=3, min_periods=1).mean()

    merged_df['deposits_volume_avg_7d'] = merged_df['deposits_volume'].rolling(window=7, min_periods=1).mean()
    merged_df['borrows_volume_avg_7d'] = merged_df['borrows_volume'].rolling(window=7, min_periods=1).mean()
    merged_df['market_cap_avg_7d'] = merged_df['market_cap'].rolling(window=7, min_periods=1).mean()
    merged_df['volumes_last_24h_avg_7d'] = merged_df['volumes_last_24h'].rolling(window=7, min_periods=1).mean()

    merged_df['daily_return'] = np.log(merged_df['price'] / merged_df['price'].shift(1))
    merged_df['volatility_3day'] = merged_df['daily_return'].rolling(window=3, min_periods=1).std()
    merged_df['volatility_7day'] = merged_df['daily_return'].rolling(window=7, min_periods=1).std()

    merged_df = merged_df[merged_df['day'] >= earliest_day]

    merged_df['liquidations'] = np.where(merged_df['debt_value_USD'] == 0, 0, 1)
    merged_df = merged_df.drop(columns=['token_debt', 'debt_amount','debt_value_USD', 'debt_current_value_USD'])

    return merged_df

@task(name=f'Train Test Split')
def train_test_split(merged_df, test_size = 40):
    last_day = merged_df.day.max()

    test_set_start = last_day - pd.DateOffset(days=test_size)
    test_set = merged_df[merged_df['day'] >= test_set_start ]
    train_set = merged_df[merged_df['day'] < test_set_start ]

    return train_set, test_set


def minmax_fit_scale(columns, df):

    scaler = MinMaxScaler()

    scaled_df = df.copy()
    scaled_df[columns] = scaler.fit_transform(scaled_df[columns])
    return scaled_df, scaler 


def minmax_scale(columns, df, scaler):

    scaled_df = df.copy()
    scaled_df[columns] = scaler.transform(scaled_df[columns])
    return scaled_df

@task(name=f'Preprocessing')
def preprocess(train_set, test_set):

    columns_to_scale = ['deposits_volume', 'borrows_volume', 'price', 'market_cap', 'volumes_last_24h',
                        'deposits_volume_avg_3d', 'borrows_volume_avg_3d', 'market_cap_avg_3d', 'volumes_last_24h_avg_3d',
                        'deposits_volume_avg_7d', 'borrows_volume_avg_7d', 'market_cap_avg_7d', 'volumes_last_24h_avg_7d', 'volatility_3day', 'volatility_7day']

    train_set_scaled,scaler = minmax_fit_scale(columns_to_scale, train_set)
    test_set_scaled = minmax_scale(columns_to_scale,test_set, scaler)

    return train_set_scaled, test_set_scaled

@task(name=f'7-3 Day Split')
def split_7_3(train_set_scaled, test_set_scaled):
    X3_train = train_set_scaled[['deposits_volume', 'borrows_volume', 'price', 'market_cap', 'volumes_last_24h', 'deposits_volume_avg_3d', 'borrows_volume_avg_3d', 'market_cap_avg_3d', 'volumes_last_24h_avg_3d', 'volatility_3day']]

    X7_train = train_set_scaled[['deposits_volume', 'borrows_volume', 'price', 'market_cap', 'volumes_last_24h', 'deposits_volume_avg_7d', 'borrows_volume_avg_7d', 'market_cap_avg_7d', 'volumes_last_24h_avg_7d', 'volatility_7day']]

    Y_train = train_set_scaled[['liquidations']]

    X3_test = test_set_scaled[['deposits_volume', 'borrows_volume', 'price', 'market_cap', 'volumes_last_24h', 'deposits_volume_avg_3d', 'borrows_volume_avg_3d', 'market_cap_avg_3d', 'volumes_last_24h_avg_3d', 'volatility_3day']]

    X7_test = test_set_scaled[['deposits_volume', 'borrows_volume', 'price', 'market_cap', 'volumes_last_24h', 'deposits_volume_avg_7d', 'borrows_volume_avg_7d', 'market_cap_avg_7d', 'volumes_last_24h_avg_7d', 'volatility_7day']]

    Y_test = test_set_scaled[['liquidations']]

    return X3_train, X7_train, Y_train, X3_test, X7_test, Y_test

@task(name=f'Model Creation')
def model_creation():
    class FeedForwardNN(nn.Module):
        def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
            super(FeedForwardNN, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size1)
            self.relu = nn.ReLU()   
            self.fc2 = nn.Linear(hidden_size1, hidden_size2)
            self.relu = nn.ReLU()
            self.fc3 = nn.Linear(hidden_size2, output_size)
            self.sigmoid = nn.Sigmoid() 
            
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.sigmoid(self.fc3(x))
            return x

    input_size = 10  
    hidden_size1 = 32
    hidden_size2 = 16
    output_size = 1  

    model_3day = FeedForwardNN(input_size, hidden_size1, hidden_size2, output_size)
    model_7day = FeedForwardNN(input_size, hidden_size1, hidden_size2, output_size)

    return model_3day, model_7day

@task(name=f'Model Training with K-folds cross validation')
def train_model(model, X, Y, num_epochs, batch_size, num_folds):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    Y_tensor = torch.tensor(Y.values, dtype=torch.float32)
    
    dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    kf = KFold(n_splits=num_folds, shuffle=False)
    fold = 1

    cv_errors = []
    
    for train_index, val_index in kf.split(X):
        print(f"Fold {fold}/{num_folds}")
        
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        Y_train, Y_val = Y.iloc[train_index], Y.iloc[val_index]
        
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
        Y_val_tensor = torch.tensor(Y_val.values, dtype=torch.float32)
        
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, Y_val_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        fold_errors = []
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        fold_errors.append(val_loss)  # Save validation loss for the fold
        cv_errors.append(fold_errors)  # Save fold errors to cross-validation errors array
        fold += 1
    
    return cv_errors

@task(name=f'Model Training and Evaluation')
def train_and_evaluate(model, X_train, Y_train, X_test, Y_test, num_epochs = 30, batch_size = 32):

    # Train the model with X_train
    cv = train_model(model, X_train, Y_train, num_epochs, batch_size, num_folds=5)

    # Set the model to evaluation mode
    model.eval()

    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    pred = model(X_test_tensor)


    # Convert the predictions to binary values
    pred_binary = (pred >= 0.5).float()

    # Convert the actual values to binary values
    Y_test_binary = torch.tensor(Y_test.values, dtype=torch.float32)

    # Calculate the metrics for X_test
    accuracy = accuracy_score(Y_test_binary, pred_binary)
    precision = precision_score(Y_test_binary, pred_binary)
    recall = recall_score(Y_test_binary, pred_binary)
    f1 = f1_score(Y_test_binary, pred_binary)
    cv_avg = np.mean(cv)
    cv_std = np.std(cv)

    return {'accuracy': accuracy,'precision': precision,'recall': recall,
            'f1': f1,'cv': cv,'cv_avg': cv_avg,'cv_std': cv_std}

@task(name=f'Export to ONNX')
def onnx_export(model, filename, input_size = 10):
    dummy_input = torch.randn(1, input_size)
    onnx.export(model, dummy_input, filename, opset_version=11)
    print(f"{filename} exported to ONNX successfully!")

@action(name=f'Model Development', log_prints=True )
def develop_model():

    merged_df, earliest_day = load_and_df_processing()
    merged_df = feature_engineering(merged_df, earliest_day)
    train_set, test_set = train_test_split(merged_df)
    train_test_scaled, test_set_scaled = preprocess(train_set, test_set)
    X3_train, X7_train, Y_train, X3_test, X7_test, Y_test = split_7_3(train_test_scaled, test_set_scaled)
    np.save('data_array.npy', X7_test.iloc[0])
    model_3day, model_7day = model_creation()
    results_3day = train_and_evaluate(model_3day, X3_train, Y_train, X3_test, Y_test)
    results_7day = train_and_evaluate(model_7day, X7_train, Y_train, X7_test, Y_test)
    onnx_export(model_3day,"model_3day.onnx")
    onnx_export(model_7day,"model_7day.onnx")


if __name__ == "__main__":
    action_deploy = Action(entrypoint=develop_model, name="aave-liquidation_model_development-action")
    action_deploy.serve(name="aave-liquidation-model")