use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor};
use orion::numbers::{FP16x16, FixedTrait};

/// Calculates the mean of a given 1D tensor.
fn calculate_mean(tensor_data: Tensor<FP16x16>) -> FP16x16 {
    let tensor_size = FixedTrait::<FP16x16>::new_unscaled(tensor_data.data.len(), false);

    let cumulated_sum = tensor_data.cumsum(0, Option::None(()), Option::None(()));
    let sum_result = cumulated_sum.data[tensor_data.data.len() - 1];
    let mean = *sum_result / tensor_size;

    return mean;
}

/// Calculates the deviation of each element from the mean of the provided 1D tensor.
fn deviation_from_mean(tensor_data: Tensor<FP16x16>) -> Tensor<FP16x16> {
    let mean_value = calculate_mean(tensor_data);

    let mut tensor_shape = array::ArrayTrait::new();
    tensor_shape.append(tensor_data.data.len());

    let mut deviation_values = array::ArrayTrait::new();

    let mut i: u32 = 0;
    loop {
        if i >= tensor_data.data.len() {
            break ();
        }
        let distance_from_mean = *tensor_data.data.at(i) - mean_value;
        deviation_values.append(distance_from_mean);
        i += 1;
    };

    let distance_from_mean_tensor = TensorTrait::<FP16x16>::new(
        tensor_shape.span(), deviation_values.span()
    );

    return distance_from_mean_tensor;
}


/// Calculates the beta value for linear regression.
fn compute_beta(x_values: Tensor<FP16x16>, y_values: Tensor<FP16x16>) -> FP16x16 {
    let x_deviation = deviation_from_mean(x_values);
    let y_deviation = deviation_from_mean(y_values);

    let x_y_covariance = x_deviation.matmul(@y_deviation);
    let x_variance = x_deviation.matmul(@x_deviation);

    let beta_value = *x_y_covariance.data.at(0) / *x_variance.data.at(0);

    return beta_value;
}

/// Calculates the intercept for linear regression.
fn compute_intercept(
    beta_value: FP16x16, x_values: Tensor<FP16x16>, y_values: Tensor<FP16x16>
) -> FP16x16 {
    let x_mean = calculate_mean(x_values);
    let y_mean = calculate_mean(y_values);

    let mx = beta_value * x_mean;
    let intercept = y_mean - mx;

    return intercept;
}

/// Predicts the y values using the provided x values and computed beta and intercept.
fn predict_y_values(
    beta_value: FP16x16, x_values: Tensor<FP16x16>, y_values: Tensor<FP16x16>
) -> Tensor<FP16x16> {
    let beta = compute_beta(x_values, y_values);
    let intercept = compute_intercept(beta_value, x_values, y_values);

    //create a tensor to hold all the y_pred values
    let mut y_pred_shape = array::ArrayTrait::new();
    y_pred_shape.append(y_values.data.len());

    let mut y_pred_vals = array::ArrayTrait::new();

    let mut i: u32 = 0;
    loop {
        if i >= y_values.data.len() {
            break ();
        }
        // (*x_values.data.at(i)).print();
        let predicted_value = beta * *x_values.data.at(i) + intercept;
        y_pred_vals.append(predicted_value);
        i += 1;
    };

    let y_pred_tensor = TensorTrait::<FP16x16>::new(y_pred_shape.span(), y_pred_vals.span());

    return y_pred_tensor;
}


/// Calculates the mean squared error between the true y values and the predicted y values.
fn compute_mse(y_values: Tensor<FP16x16>, y_pred_values: Tensor<FP16x16>) -> FP16x16 {
    let mut squared_diff_shape = array::ArrayTrait::new();
    squared_diff_shape.append(y_values.data.len());

    let mut squared_diff_vals = array::ArrayTrait::new();

    let mut i: u32 = 0;
    loop {
        if i >= y_values.data.len() {
            break ();
        }
        let diff = *y_values.data.at(i) - *y_pred_values.data.at(i);
        let squared_diff = diff * diff;
        squared_diff_vals.append(squared_diff);
        i += 1;
    };

    let squared_diff_tensor = TensorTrait::<FP16x16>::new(
        squared_diff_shape.span(), squared_diff_vals.span()
    );

    let mse = calculate_mean(squared_diff_tensor);

    return mse;
}

/// Calculates the R squared score.
fn calculate_r_score(y_values: Tensor<FP16x16>, y_pred_values: Tensor<FP16x16>) -> FP16x16 {
    let mean_y_value = calculate_mean(y_values);

    // creating the appropriate tensor shapes and empty arrays to populate values into
    let mut squared_diff_shape = array::ArrayTrait::new();
    squared_diff_shape.append(y_values.data.len());
    let mut squared_diff_vals = array::ArrayTrait::new();

    let mut squared_mean_diff_shape = array::ArrayTrait::new();
    squared_mean_diff_shape.append(y_values.data.len());
    let mut squared_mean_diff_vals = array::ArrayTrait::new();

    let mut i: u32 = 0;
    loop {
        if i >= y_values.data.len() {
            break ();
        }
        let diff_pred = *y_values.data.at(i) - *y_pred_values.data.at(i);
        let squared_diff = diff_pred * diff_pred;
        squared_diff_vals.append(squared_diff);

        let diff_mean = *y_values.data.at(i) - mean_y_value;
        let squared_mean_diff = diff_mean * diff_mean;
        squared_mean_diff_vals.append(squared_mean_diff);
        i += 1;
    };

    let squared_diff_tensor = TensorTrait::<FP16x16>::new(
        squared_diff_shape.span(), squared_diff_vals.span()
    );
    let squared_mean_diff_tensor = TensorTrait::<FP16x16>::new(
        squared_mean_diff_shape.span(), squared_mean_diff_vals.span()
    );

    let sum_squared_diff = squared_diff_tensor.cumsum(0, Option::None(()), Option::None(()));
    let sum_squared_mean_diff = squared_mean_diff_tensor
        .cumsum(0, Option::None(()), Option::None(()));

    let r_score = FixedTrait::new_unscaled(1, false)
        - *sum_squared_diff.data.at(y_values.data.len() - 1)
            / *sum_squared_mean_diff.data.at(y_values.data.len() - 1);

    return r_score;
}

