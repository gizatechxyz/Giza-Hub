
use orion::operators::tensor::{
    Tensor, TensorTrait, FP16x16Tensor, U32Tensor, U32TensorAdd, FP16x16TensorSub, FP16x16TensorAdd,
    FP16x16TensorDiv, FP16x16TensorMul
};
use orion::numbers::{FP16x16, FixedTrait};
use linear_regresion::data_preprocessing::{Dataset, DatasetTrait};
use linear_regresion::helper_functions::{
    get_tensor_data_by_row, transpose_tensor, calculate_mean, calculate_r_score,
    normalize_user_x_inputs, rescale_predictions
};

#[derive(Copy, Drop)]
struct LinearRegressionModel {
    gradient: Tensor<FP16x16>,
    bias: Tensor<FP16x16>
}

#[generate_trait]
impl RegressionOperation of LinearRegressionModelTrait {
    fn predict(ref self: LinearRegressionModel, x_input: Tensor<FP16x16>) -> Tensor<FP16x16> {
        let gradient = self.gradient;
        let bias = self.bias;
        let mut prediction = (gradient * x_input) + bias;
        return prediction;
    }
}

fn LinearRegression(dataset: Dataset) -> LinearRegressionModel {
    let gradient = compute_gradient(dataset);
    let bias = compute_bias(dataset);
    return LinearRegressionModel { gradient, bias };
}

// computes the mean of a given 1D tensor and outputs result as tensor
fn compute_mean(tensor_data: Tensor<FP16x16>) -> Tensor<FP16x16> {
    let tensor_size = FixedTrait::<FP16x16>::new_unscaled(tensor_data.data.len(), false);
    let cumulated_sum = tensor_data.cumsum(0, Option::None(()), Option::None(()));
    let sum_result = cumulated_sum.data[tensor_data.data.len() - 1];
    let mean = *sum_result / tensor_size;
    let mut result_tensor = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![mean].span());
    return result_tensor;
}


/// Calculates the deviation of each element from the mean of the provided 1D tensor.
fn deviation_from_mean(tensor_data: Tensor<FP16x16>) -> Tensor<FP16x16> {
    let mut tensor_data_ = tensor_data;
    let mean_value = calculate_mean(tensor_data);
    let mut tensor_shape = array::ArrayTrait::new();
    tensor_shape.append(tensor_data.data.len());
    let mut deviation_values = array::ArrayTrait::new();

    let mut i: u32 = 0;

    loop {
        match tensor_data_.data.pop_front() {
            Option::Some(tensor_val) => {
                let distance_from_mean = *tensor_val - mean_value;
                deviation_values.append(distance_from_mean);
                i += 1;
            },
            Option::None(_) => { break; }
        };
    };
    let distance_from_mean_tensor = TensorTrait::<
        FP16x16
    >::new(tensor_shape.span(), deviation_values.span());

    return distance_from_mean_tensor;
}

/// Calculates the beta value for linear regression.
fn compute_gradient(dataset: Dataset) -> Tensor<FP16x16> {
    let x_deviation = deviation_from_mean(dataset.x_values);
    let y_deviation = deviation_from_mean(dataset.y_values);

    let x_y_covariance = x_deviation.matmul(@y_deviation);
    let x_variance = x_deviation.matmul(@x_deviation);

    let beta_value = x_y_covariance / x_variance;

    return beta_value;
}


/// Calculates the intercept for linear regression.
fn compute_bias(dataset: Dataset) -> Tensor<FP16x16> {
    let x_mean = compute_mean(dataset.x_values);
    let y_mean = compute_mean(dataset.y_values);
    let gradient = compute_gradient(dataset);
    let mx = gradient * x_mean;
    let intercept = y_mean - mx;
    return intercept;
}

