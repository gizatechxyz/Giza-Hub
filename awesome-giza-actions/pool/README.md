## Time Series Forecasting with WaveNet for Pool Volume and Cryptocurrency Trading
### Problem Setting

The project aims to develop a WaveNet model for time series forecasting, specifically for predicting pool volumes using time-series data and cryptocurrency trading volume in the financial market. The WaveNet architecture is chosen for its ability to capture long-range dependencies in sequential data effectively. The ZKML (Zero-Knowledge Machine Learning) framework is utilized to ensure the verifiability of the model predictions, which is crucial for applications in financial markets. Interested readers can refer to the WaveNet paper by van den Oord et al. (2016) for a detailed understanding of the architecture.

### Project Installation

To reproduce the project, follow these steps:

    Clone the repository:


```
git clone https://github.com/gizatechxyz/Giza-Hub.git

    Navigate to the project directory:
```
```
cd awesome-giza-actions/pool
```
    Install dependencies:

```
poetry shell
poetry install

```

## Overview of Model Development

### The model development process involves several key steps:

#### Defining Model Hyperparameters:
This task defines the hyperparameters such as input shape, number of filters, number of residual blocks, kernel size, number of classes, batch size, and window size required for the WaveNet model.

```
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
```

#### Loading Data:
The dataset, containing daily trade volume from a cryptocurrency exchange, is loaded using the DatasetsLoader module from Giza dataset. Additionally, pool volume data is loaded from Polar into a DataFrame.

```
@task(name="Load Data")
def load_data():
    loader = DatasetsLoader()
    # Load data from Polar into a DataFrame
    df_polar = loader.load('balancer-daily-trade-volume')
    return df_polar
```

#### Preprocessing Data: 
Data preprocessing involves extracting relevant features, performing one-hot encoding for categorical variables, standardizing numerical features, and extracting temporal features such as day of the week, month, and year for both datasets.

```
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
```

#### Creating Sequences: 
Time series data is transformed into input-output sequences suitable for training the WaveNet model.

```
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
```

#### Preparing Datasets: 
The datasets are split into training, validation, and testing sets for model training and evaluation.

```
@task(name="Prepare Datasets")
def prepare_datasets(df_encoded):
    print("Prepare dataset...")

    # Splitting into training and testing sets (80% train, 20% test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(df_encoded.drop(columns=['trading_volume_usd']), df_encoded['trading_volume_usd'], test_size=0.2, random_state=42)

    # Splitting the training set into training and validation sets (80% train, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    print("✅ Datasets prepared successfully")

    return X_train, y_train, X_test, y_test
```

#### Building the Model:
The WaveNet model architecture is constructed using TensorFlow's Keras API, comprising convolutional layers with dilated causal convolutions and skip connections.

```
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
```

#### Test the model
This task defines a function to evaluate the performance of the trained model on the test dataset. It computes the loss and mean absolute error (MAE) metrics and prints the results.

```
@task(name="Test Model")
def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the test dataset.

    Args:
        model: The trained model to be evaluated.
        X_test: Input features of the test dataset.
        y_test: Target labels of the test dataset.
    """
    # Evaluate the model
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print("Test Loss:", test_loss)
    print("Test MAE:", test_mae)
```

#### Converting to ONNX:
The trained TensorFlow model is converted to the ONNX format for verifiability using the tf2onnx library.

```
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
```

#### Prediction with ONNX
This task defines a function to make predictions using the ONNX format of the trained WaveNet model. The input data is fed to the model, and the predictions are returned after reshaping the output.

```
@task(name="Prediction with ONNX")
def prediction(X_val):
    model = GizaModel(model_path="./wavenet.onnx")

    # Ensure the input data matches the expected shape (1, 30, 14)
    if X_val.shape != (1, 30, 14):
        print("Invalid input shape. Expected shape: (1, 30, 14)")
        return None

    result = model.predict(input_feed={"input_1": X_val}, verifiable=False)

    return result.reshape(-1, 1)  # Reshape the output to (30, 1)
```

#### Predictions with Cairo
This task defines a function to make predictions using the CAIRO format of the trained WaveNet model. The input data is fed to the model, and the predictions along with the request ID are returned.

```
@task(name="Prediction with CAIRO")
def prediction(X_val, model_id, version_id):
    model = GizaModel(id=model_id, version=version_id)

    # Ensure the input data matches the expected shape (1, 30, 14)
    if X_val.shape != (1, 30, 14):
        print("Invalid input shape. Expected shape: (1, 30, 14)")
        return None

    (result, request_id) = model.predict(input_feed={"input_1": X_val}, verifiable=True, output_dtype='Tensor<FP16x16>'
  )

    return result, request_id
```

#### Download and verify the proof job
This command downloads the proof job associated with a specific model deployment from the Giza platform
```
giza deployments download-proof --model-id <MODEL_ID> --version-id <VERSION_ID> --deployment-id <DEPLOYMENT_ID> --proof-id <PROOF_ID> --output-path <OUTPUT_PATH>
```

While this command verifies the downloaded proof file to ensure the integrity and authenticity of the model predictions.
```
giza verify --proof PATH_OF_THE_PROOF
```

### Model Performance

The model's performance is crucial in assessing its effectiveness in predicting cryptocurrency trading volumes. Two primary metrics used for evaluation are Mean Squared Error (MSE) and Mean Absolute Error (MAE).

Here are the performance metrics obtained from testing the model:

    Test Loss: 0.9989119172096252
    Test MAE: 0.9989119172096252

These values provide insights into how well the model generalizes to unseen data. A lower test loss and MAE indicate better performance, as they signify that the model's predictions are closer to the actual values.

Possible improvements to enhance model performance may include adjusting hyperparameters such as learning rate, batch size, or introducing additional layers to capture more complex patterns in the data. Additionally, exploring alternative architectures or incorporating external data sources could also lead to better predictions. Regular monitoring and iterative refinement of the model are essential for achieving optimal performance in forecasting cryptocurrency trading volumes.

### Giza Integration

In addition to using Giza Dataset, Giza CLI & Actions are used to make the WaveNet model verifiable. This involves defining tasks and actions within the Giza framework to ensure the reproducibility and verifiability of the model predictions.



## Tech Stack

    Giza Actions SDK
    Giza cli
    Giza Virtual Environment
    Giza Dataset
    WaveNet
    Jupyter Notebook
    Tensorflow
    Poetry
    Cairo
    EZKL
    ONNX
