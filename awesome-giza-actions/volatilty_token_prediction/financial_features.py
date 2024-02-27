import numpy as np

def calculate_rsi(series, window=7):
    """
    Calculates the Relative Strength Index (RSI) of a given price series.
    
    Parameters:
    - series (pd.Series): The price series to calculate RSI for.
    - window (int): The lookback period for calculating RSI, default is 7 days.
    
    Returns:
    - pd.Series: The RSI values.
    """
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, short_window=12, long_window=26, signal_window=9):
    """
    Calculates the Moving Average Convergence Divergence (MACD) and its signal line for a price series.
    
    Parameters:
    - series (pd.Series): The price series.
    - short_window (int): The period for the short-term EMA.
    - long_window (int): The period for the long-term EMA.
    - signal_window (int): The period for the signal line.
    
    Returns:
    - tuple of pd.Series: (MACD line, signal line).
    """
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(series, window=20):
    """
    Calculates the Bollinger Bands for a price series.
    
    Parameters:
    - series (pd.Series): The price series.
    - window (int): The moving average window size.
    
    Returns:
    - tuple of pd.Series: (upper band, lower band).
    """
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    bollinger_up = sma + (std * 2)
    bollinger_down = sma - (std * 2)
    return bollinger_up, bollinger_down

def calculate_future_volatility(df, window=7):
    """
    Calculates future volatility of a financial instrument based on log returns.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the 'returns' column.
    - window (int): The window size to calculate rolling standard deviation.
    
    Returns:
    - pd.Series: The future volatility values.
    """

    df['log_returns'] = np.log(1 + df['returns'])
    vol = df['log_returns'].rolling(window=window).std().shift(-window)
    return vol

def add_mean_percentage_changes(df, window=7):
    """
    Adds columns to the DataFrame representing the percentage change over a specified window for price, volume, and market cap.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame with 'price', 'volumes_last_24h', and 'market_cap' columns.
    - window (int): The period over which the percentage change is calculated.
    
    Returns:
    - pd.DataFrame: The DataFrame with added percentage change columns.
    """
    df[f'price_pct_change_{window}d'] = df['price'].pct_change(periods=window)
    df[f'volume_pct_change_{window}d'] = df['volumes_last_24h'].pct_change(periods=window)
    df[f'market_cap_pct_change_{window}d'] = df['market_cap'].pct_change(periods=window)
    return df

def add_ma(df, window=7):
    """
    Adds columns to the DataFrame representing the moving average over a specified window for price, volume, and market cap.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame with 'price', 'volumes_last_24h', and 'market_cap' columns.
    - window (int): The window size for the moving average.
    
    Returns:
    - pd.DataFrame: The DataFrame with added moving average columns.
    """

    df[f'price_ma_{window}d'] = df['price'].rolling(window=window).mean()
    df[f'volume_ma_{window}d'] = df['volumes_last_24h'].rolling(window=window).mean()
    df[f'market_cap_ma_{window}d'] = df['market_cap'].rolling(window=window).mean()
    return df

def add_deltas(df, window=7):
    """
    Adds columns to the DataFrame representing the difference (delta) over a specified window for price, volume, and market cap.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame with 'price', 'volumes_last_24h', and 'market_cap' columns.
    - window (int): The period over which the difference is calculated.
    
    Returns:
    - pd.DataFrame: The DataFrame with added delta columns.
    """
    df[f'price_delta_{window}d'] = df['price'].diff(periods=window)
    df[f'volume_delta_{window}d'] = df['volumes_last_24h'].diff(periods=window)
    df[f'market_cap_delta_{window}d'] = df['market_cap'].diff(periods=window)
    return df

def add_momentum(df, window=7):
    """
    Adds a momentum column to the DataFrame calculated over a specified window for the price.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame with a 'price' column.
    - window (int): The window size for calculating momentum.
    
    Returns:
    - pd.DataFrame: The DataFrame with an added momentum column.
    """
    df[f'momentum_{window}d'] = df['price'] / df['price'].shift(window) - 1
    return df

def add_past_volatility(df, window=7):
    """
    Adds a volatility column to the DataFrame calculated as the standard deviation of returns over a specified window.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame with a 'returns' column.
    - window (int): The window size for calculating volatility.
    
    Returns:
    - pd.DataFrame: The DataFrame with an added volatility column.
    """

    df[f'price_volatility_{window}d'] = df['returns'].rolling(window=window).std()
    return df

def add_min_max_relatives(df, window=7):
    """
    Adds columns to the DataFrame representing the minimum and maximum price over a specified window.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame with a 'price' column.
    - window (int): The window size to calculate the min and max prices.
    
    Returns:
    - pd.DataFrame: The DataFrame with added min and max price columns.
    """
    df[f'price_max_{window}d'] = df['price'].rolling(window=window).max()
    df[f'price_min_{window}d'] = df['price'].rolling(window=window).min()
    return df

def add_percentage_range(df, window=7):
    """
    Adds a column to the DataFrame representing the percentage range between the min and max prices over a specified window.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame with 'price', 'price_min_{window}d', and 'price_max_{window}d' columns.
    - window (int): The window size used in calculating the min and max prices.
    
    Returns:
    - pd.DataFrame: The DataFrame with an added percentage range column.
    """
    df[f'price_range_pct_{window}d'] = (df[f'price_max_{window}d'] - df[f'price_min_{window}d']) / df['price']
    return df

def add_financial_features(df):
    """
    Adds a comprehensive set of financial features to the DataFrame, including moving averages, momentum, volatility, and others.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame with 'price', 'volumes_last_24h', 'market_cap', and 'date' columns.
    
    Returns:
    - pd.DataFrame: The DataFrame enriched with various financial features.
    """
    df.set_index('date', inplace=True)
    df['returns'] = df['price'].pct_change()

    for window in [7, 15, 30]:
        df = add_mean_percentage_changes(df, window)
        df = add_ma(df, window)
        df = add_momentum(df, window)
        df = add_past_volatility(df, window)
        df = add_min_max_relatives(df, window)
        df = add_percentage_range(df, window)

    df['rsi_14d'] = calculate_rsi(df['price'], window=14)
    df['macd'], df['macd_signal_line'] = calculate_macd(df['price'])
    df['bollinger_up_20d'], df['bollinger_down_20d'] = calculate_bollinger_bands(df['price']) 
    return df