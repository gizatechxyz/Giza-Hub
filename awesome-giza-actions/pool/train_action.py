import numpy as np
from giza_actions.action import Action, action
from giza_actions.task import task
from giza_datasets import DatasetsLoader
import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
import pandas as pd
import polars as pl
import tf2onnx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import Sequence
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Activation, Add
import onnx
from typing import List

@task(name="Define Model Hyperparameters")
def define_model_hyperparameters():
    selected_columns = ['trading_volume_usd', 'blockchain_arbitrum', 'blockchain_avalanche_c', 'blockchain_base', 'blockchain_ethereum', 'blockchain_gnosis', 'blockchain_optimism', 'blockchain_polygon', 'token_pair_wstETH-wUSDM', 'token_pair_xSNXa-YFI', 'token_pair_yCURVE-YFI', 'day_of_week', 'month', 'year']
    window_size = 30
    input_shape = (window_size, len(selected_columns))
    num_filters = 32
    num_blocks = 5
    kernel_size = 8
    num_classes = 1  # Assuming regression task, change to number of classes for classification task
    batch_size = 32

    return input_shape, num_filters, num_blocks, kernel_size, num_classes, batch_size, window_size

@task(name="Load Data")
def load_data():
    loader = DatasetsLoader()
    # Load data from Polar into a DataFrame
    df_polar = loader.load('balancer-daily-trade-volume')
    return df_polar

@task(name="Preprocess Data")
def preprocess_data(df_polar):
    # Extracting data from the Polar DataFrame
    data = {
        'day': df_polar['day'],
        'pool_id': df_polar['pool_id'],
        'blockchain': df_polar['blockchain'],
        'token_pair': df_polar['token_pair'],
        'trading_volume_usd': df_polar['trading_volume_usd']
    }

    # Creating a new Pandas DataFrame
    df_pandas = pd.DataFrame(data)

    # Perform one-hot encoding for categorical variables
    df_encoded = pd.get_dummies(df_pandas, columns=['blockchain', 'token_pair'])

    # Initialize StandardScaler
    standard_scaler = StandardScaler()

    # Perform Standardization on numerical features
    df_encoded[['trading_volume_usd']] = standard_scaler.fit_transform(df_encoded[['trading_volume_usd']])

    # Convert 'day' column to datetime format
    df_encoded['day'] = pd.to_datetime(df_encoded['day'])

    # Extract relevant features: day of the week, month, and year
    df_encoded['day_of_week'] = df_encoded['day'].dt.dayofweek
    df_encoded['month'] = df_encoded['day'].dt.month
    df_encoded['year'] = df_encoded['day'].dt.year

    return df_encoded

@task(name="Create Sequences")
def create_sequences(df_encoded, window_size):
    # Calculate the total number of data points
    total_data_points = df_encoded.shape[0]

    # Calculate the total number of sequences
    total_sequences = total_data_points - window_size + 1

    # Select only necessary columns from the DataFrame
    selected_columns = ['trading_volume_usd', 'blockchain_arbitrum', 'blockchain_avalanche_c', 'blockchain_base', 'blockchain_ethereum', 'blockchain_gnosis', 'blockchain_optimism', 'blockchain_polygon', 'token_pair_wstETH-wUSDM', 'token_pair_xSNXa-YFI', 'token_pair_yCURVE-YFI', 'day_of_week', 'month', 'year']
    df_selected = df_encoded[selected_columns]

    # Slide a window of this length across your time-series data
    sequences_input = []
    sequences_target = []

    for i in range(total_sequences):
        # Extract the historical data points as the input sequence
        input_sequence = df_selected.iloc[i : i + window_size].values
        sequences_input.append(input_sequence)
        # Extract the next data point as the target for prediction
        target = df_selected.iloc[i + window_size - 1, 2]  
        sequences_target.append(target)

    # Convert lists to numpy arrays
    sequences_input = np.array(sequences_input)
    sequences_target = np.array(sequences_target)

    # Reshape the target sequences to match the shape of the input sequences
    sequences_target = sequences_target.reshape(-1, 1)

    sequences_input = sequences_input.astype(np.float32)
    sequences_target = sequences_target.astype(np.float32)

    return sequences_input, sequences_target



@task(name="Prepare Datasets")
def prepare_datasets(df_encoded):
    print("Prepare dataset...")

    # Splitting into training and testing sets (80% train, 20% test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(df_encoded.drop(columns=['trading_volume_usd']), df_encoded['trading_volume_usd'], test_size=0.2, random_state=42)

    # Splitting the training set into training and validation sets (80% train, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    print("✅ Datasets prepared successfully")

    return X_train, y_train, X_test, y_test

@task(name=" Data Generator")
class DataGenerator(Sequence):
    def __init__(self, sequences_input, sequences_target, batch_size):
        self.sequences_input = sequences_input
        self.sequences_target = sequences_target
        self.batch_size = batch_size

    def __len__(self):
        return len(self.sequences_input) // self.batch_size

    def __getitem__(self, idx):
        batch_inputs = self.sequences_input[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_targets = self.sequences_target[idx * self.batch_size : (idx + 1) * self.batch_size]
        return np.array(batch_inputs), np.array(batch_targets)

@task(name="Test Model")
def evaluate_model(model, X_test, y_test):
    # Evaluate the model
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print("Test Loss:", test_loss)
    print("Test MAE:", test_mae)


@task(name="Train model")
def build_wavenet(input_shape, num_filters, num_blocks, kernel_size, num_classes):
    def residual_block(x, filters, kernel_size, dilation_rate):
        # Dilated causal convolution
        conv = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal')(x)
        conv = Activation('relu')(conv)

        # 1x1 convolution to adjust the number of filters
        conv = Conv1D(filters=filters, kernel_size=1)(conv)

        # Skip connection
        x = Add()([x, conv])
        return x

    inputs = Input(shape=input_shape)

    # Initial convolution block
    x = Conv1D(filters=num_filters, kernel_size=1, padding='causal')(inputs)

    # Dilated causal convolutions
    for i in range(num_blocks):
        dilation_rate = 2 ** i
        x = residual_block(x, num_filters, kernel_size, dilation_rate)

    # Output layer
    outputs = Conv1D(filters=num_classes, kernel_size=1, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)


    print("✅ Model trained successfully")
    return model


@task(name="Convert To ONNX")
def convert_to_onnx(model, onnx_file_path):
    """
    Convert a TensorFlow model to ONNX format and save it.

    Args:
        model (tf.keras.Model): The TensorFlow model to convert.
        onnx_file_path (str): The path to save the ONNX model file.
    """
    # Convert TensorFlow model to ONNX
    onnx_model, _ = tf2onnx.convert.from_keras(model)

    # Save the ONNX model
    with open(onnx_file_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"Model has been converted to ONNX and saved as {onnx_file_path}")

@action(name="Action: Convert To ONNX", log_prints=True)
def execution():

    input_shape, num_filters, num_blocks, kernel_size, num_classes, batch_size, window_size = define_model_hyperparameters()
    df_polar = load_data()
    df_encoded = preprocess_data(df_polar)
    sequences_input, sequences_target = create_sequences(df_encoded, window_size)


    # Prepare datasets
    X_train, y_train, X_test, y_test = prepare_datasets(df_encoded)

    # Create data loaders
    data_generator = DataGenerator(sequences_input, sequences_target, batch_size)

    # Train the model
    # Build the WaveNet model
    model = build_wavenet(input_shape, num_filters, num_blocks, kernel_size, num_classes)

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Print model summary
    model.summary()


    data_generator = DataGenerator(sequences_input, sequences_target, batch_size)

    # Train the model using the fit method
    model.fit(data_generator, epochs=1)
    # Evaluate the model
   # evaluate_model(model, X_test, y_test)


    # Convert to ONNX
    onnx_file_path = "wavenet.onnx"
    convert_to_onnx(model, onnx_file_path)



# Create an Action object and serve it
if __name__ == "__main__":
    action_deploy = Action(entrypoint=execution, name="pytorch-trade-action")
    action_deploy.serve(name="pytorch-trade-deployment")
