

use debug::PrintTrait;
use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{
    Tensor, TensorTrait, FP16x16Tensor, U32Tensor, U32TensorAdd, FP16x16TensorSub, FP16x16TensorAdd,
    FP16x16TensorDiv, FP16x16TensorMul
};

use orion::numbers::{FP16x16, FixedTrait};

// retrieves row data by index in a 2D tensor
fn get_tensor_data_by_row(tensor_data: Tensor<FP16x16>, row_index: u32,) -> Tensor<FP16x16> {
    let column_len = *tensor_data.shape.at(1); //13
    // crete new array
    let mut result = ArrayTrait::<FP16x16>::new();
    // loop through the x values and append values 
    let mut i: u32 = 0;
    loop {
        if i >= column_len {
            break ();
        }
        result.append(tensor_data.at(indices: array![row_index, i].span()));
        i += 1;
    };
    let resultant_tensor = TensorTrait::<
        FP16x16
    >::new(array![column_len].span(), data: result.span());
    return resultant_tensor;
}


// transposes tensor
fn transpose_tensor(tensor_data: Tensor<FP16x16>) -> Tensor<FP16x16> {
    let tensor_transposed = tensor_data.transpose(axes: array![1, 0].span());
    return tensor_transposed;
}

fn calculate_mean(tensor_data: Tensor<FP16x16>) -> FP16x16 {
    let tensor_size = FixedTrait::<FP16x16>::new_unscaled(tensor_data.data.len(), false);
    let cumulated_sum = tensor_data.cumsum(0, Option::None(()), Option::None(()));
    let sum_result = cumulated_sum.data[tensor_data.data.len() - 1];
    let mean = *sum_result / tensor_size;
    return mean;
}

// Calculates the R-Squared score between two tensors.
fn calculate_r_score(Y_values: Tensor<FP16x16>, Y_pred_values: Tensor<FP16x16>) -> FP16x16 {
    let mut Y_values_ = Y_values;
    let mean_y_value = calculate_mean(Y_values);
    // creating the appropriate tensor shapes and empty arrays to populate values into
    let mut squared_diff_shape = array::ArrayTrait::new();
    squared_diff_shape.append(Y_values.data.len());
    let mut squared_diff_vals = array::ArrayTrait::new();
    let mut squared_mean_diff_shape = array::ArrayTrait::new();
    squared_mean_diff_shape.append(Y_values.data.len());
    let mut squared_mean_diff_vals = array::ArrayTrait::new();

    let mut i: u32 = 0;

    loop {
        match Y_values_.data.pop_front() {
            Option::Some(y_value) => {
                let diff_pred = *y_value - *Y_pred_values.data.at(i);
                let squared_diff = diff_pred * diff_pred;
                squared_diff_vals.append(squared_diff);

                let diff_mean = *y_value - mean_y_value;
                let squared_mean_diff = diff_mean * diff_mean;
                squared_mean_diff_vals.append(squared_mean_diff);
                i += 1;
            },
            Option::None(_) => { break; }
        }
    };

    let squared_diff_tensor = TensorTrait::<
        FP16x16
    >::new(squared_diff_shape.span(), squared_diff_vals.span());
    let squared_mean_diff_tensor = TensorTrait::<
        FP16x16
    >::new(squared_mean_diff_shape.span(), squared_mean_diff_vals.span());
    let sum_squared_diff = squared_diff_tensor.cumsum(0, Option::None(()), Option::None(()));
    let sum_squared_mean_diff = squared_mean_diff_tensor
        .cumsum(0, Option::None(()), Option::None(()));
    let r_score = FixedTrait::new_unscaled(1, false)
        - *sum_squared_diff.data.at(Y_values.data.len() - 1)
            / *sum_squared_mean_diff.data.at(Y_values.data.len() - 1);

    return r_score;
}


// computes the x_min, x_max and x_range. Used for helping in normalizing and denormalizing user inputed values operations
fn normalize_user_x_inputs(
    x_inputs: Tensor<FP16x16>, original_x_values: Tensor<FP16x16>
) -> Tensor<FP16x16> {
    let mut x_inputs_normalized = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::new(10, false)].span());

    let mut x_min = ArrayTrait::<FP16x16>::new();
    let mut x_max = ArrayTrait::<FP16x16>::new();
    let mut x_range = ArrayTrait::<FP16x16>::new();
    let mut result = ArrayTrait::<FP16x16>::new();

    if original_x_values.shape.len() > 1 {
        let transposed_tensor = original_x_values.transpose(axes: array![1, 0].span());
        let data_len = *transposed_tensor.shape.at(0); //13
        // loop through each row calculating the min, max and range row values for each feature columns
        let mut i: u32 = 0;
        loop {
            if i >= data_len {
                break ();
            }
            let mut transposed_tensor_row = get_tensor_data_by_row(transposed_tensor, i);
            x_min.append(transposed_tensor_row.min_in_tensor());
            x_max.append(transposed_tensor_row.max_in_tensor());
            x_range
                .append(
                    transposed_tensor_row.max_in_tensor() - transposed_tensor_row.min_in_tensor()
                );
            i += 1;
        };
        let mut x_min_tensor = TensorTrait::new(shape: array![data_len].span(), data: x_min.span());
        let mut x_max_tensor = TensorTrait::new(shape: array![data_len].span(), data: x_max.span());
        let mut x_range_tensor = TensorTrait::new(
            shape: array![data_len].span(), data: x_range.span()
        );

        // for normalizing 2D user inputed feature vals
        if x_inputs.shape.len() > 1 {
            let mut j: u32 = 0;
            loop {
                if j >= *x_inputs.shape.at(0) {
                    break ();
                };
                let mut row_data = get_tensor_data_by_row(x_inputs, j);
                let mut norm_row_data = (row_data - x_min_tensor) / x_range_tensor;
                let mut k: u32 = 0;

                loop {
                    if k >= norm_row_data.data.len() {
                        break ();
                    };
                    result.append(*norm_row_data.data.at(k));
                    k += 1;
                };
                j += 1;
            };
            x_inputs_normalized =
                TensorTrait::<
                    FP16x16
                >::new(
                    array![*x_inputs.shape.at(0), *x_inputs.shape.at(1)].span(), data: result.span()
                );
        };

        // for normalizing 1D feature input
        if x_inputs.shape.len() == 1 {
            x_inputs_normalized = (x_inputs - x_min_tensor) / x_range_tensor;
        };
    }

    if original_x_values.shape.len() == 1 {
        let mut x_min_tensor = TensorTrait::<
            FP16x16
        >::new(shape: array![1].span(), data: array![original_x_values.min_in_tensor()].span());
        let mut x_max_tensor = TensorTrait::<
            FP16x16
        >::new(shape: array![1].span(), data: array![original_x_values.max_in_tensor()].span());
        let mut x_range_tensor = TensorTrait::<
            FP16x16
        >::new(
            shape: array![1].span(),
            data: array![original_x_values.max_in_tensor() - original_x_values.min_in_tensor()]
                .span()
        );
        let mut diff = ((x_inputs - x_min_tensor));
        x_inputs_normalized = ((x_inputs - x_min_tensor)) / x_range_tensor;
    };
    return x_inputs_normalized;
}


// rescales model predictions to standard format
fn rescale_predictions(
    prediction_result: Tensor<FP16x16>, y_values: Tensor<FP16x16>
) -> Tensor<FP16x16> {
    let mut rescale_predictions = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::new(10, false)].span());

    let mut y_min_array = ArrayTrait::<FP16x16>::new();
    let mut y_max_array = ArrayTrait::<FP16x16>::new();
    let mut y_range_array = ArrayTrait::<FP16x16>::new();

    let mut y_max = y_values.max_in_tensor();
    let mut y_min = y_values.min_in_tensor();
    let mut y_range = y_values.max_in_tensor() - y_values.min_in_tensor();
    // convert to tensor format for ease of math operations
    let y_min_tensor = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![y_min].span());
    let y_max_tensor = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![y_max].span());
    let y_range_tensor = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![y_range].span());

    rescale_predictions = (prediction_result * y_range_tensor) + y_min_tensor;

    return rescale_predictions;
}

