

use orion::operators::tensor::{
    Tensor, TensorTrait, FP16x16Tensor, U32Tensor, U32TensorAdd, FP16x16TensorSub, FP16x16TensorAdd,
    FP16x16TensorDiv, FP16x16TensorMul
};
use orion::numbers::{FP16x16, FixedTrait};
use linear_regresion::helper_functions::{
    get_tensor_data_by_row, transpose_tensor, calculate_mean, calculate_r_score,
    normalize_user_x_inputs, rescale_predictions
};

#[derive(Copy, Drop)]
struct Dataset {
    x_values: Tensor<FP16x16>,
    y_values: Tensor<FP16x16>,
}

#[generate_trait]
impl DataPreprocessing of DatasetTrait {
    fn normalize_dataset(ref self: Dataset) -> Dataset {
        let mut x_values = TensorTrait::<FP16x16>::new(array![1].span(), array![FixedTrait::new(0, false)].span());
        let mut y_values = TensorTrait::<FP16x16>::new(array![1].span(), array![FixedTrait::new(0, false)].span());
        // used for multiple_linear_regression_models
        if self.x_values.shape.len() > 1 {
            x_values = normalize_feature_data(self.x_values);
            y_values = normalize_label_data(self.y_values);
        }
        // used for linear_regression_models
        if self.x_values.shape.len() == 1 {
            x_values = normalize_label_data(self.x_values);
            y_values = normalize_label_data(self.y_values);
        }

        return Dataset { x_values, y_values };
    }
}

// normalizes 2D Tensor
fn normalize_feature_data(tensor_data: Tensor<FP16x16>) -> Tensor<FP16x16> {
    let mut x_min_array = ArrayTrait::<FP16x16>::new();
    let mut x_max_array = ArrayTrait::<FP16x16>::new();
    let mut x_range_array = ArrayTrait::<FP16x16>::new();
    let mut normalized_array = ArrayTrait::<FP16x16>::new();
    // transpose to change rows to be columns
    let transposed_tensor = tensor_data.transpose(axes: array![1, 0].span());
    let tensor_shape = transposed_tensor.shape;
    let tensor_row_len = *tensor_shape.at(0); // 13 
    let tensor_column_len = *tensor_shape.at(1); //50
    // loop and append max and min row values to corresponding  array
    let mut i: u32 = 0;
    loop {
        if i >= tensor_row_len {
            break ();
        }
        let mut transposed_tensor_row = get_tensor_data_by_row(transposed_tensor, i);
        x_max_array.append(transposed_tensor_row.max_in_tensor());
        x_min_array.append(transposed_tensor_row.min_in_tensor());
        x_range_array
            .append(transposed_tensor_row.max_in_tensor() - transposed_tensor_row.min_in_tensor());
        i += 1;
    };
    // convert array to tensor format for ease of math operation
    let mut x_min = TensorTrait::<
        FP16x16
    >::new(shape: array![1, tensor_row_len].span(), data: x_min_array.span());
    let mut x_range = TensorTrait::<
        FP16x16
    >::new(shape: array![1, tensor_row_len].span(), data: x_range_array.span());
    let normalized_tensor = (tensor_data - x_min) / x_range;
    return normalized_tensor;
}

// normalizes 1D tensor
fn normalize_label_data(tensor_data: Tensor<FP16x16>) -> Tensor<FP16x16> {
    let mut tensor_data_ = tensor_data;
    let mut normalized_array = ArrayTrait::<FP16x16>::new();
    let mut range = tensor_data.max_in_tensor() - tensor_data.min_in_tensor();
    // loop through tensor values normalizing and appending to new array
    let mut i: u32 = 0;

    loop {
        match tensor_data_.data.pop_front() {
            Option::Some(tensor_val) => {
                let mut diff = *tensor_val - tensor_data.min_in_tensor();
                normalized_array.append(diff / range);
                i += 1;
            },
            Option::None(_) => { break; }
        };
    };
    // convert normalized array values to tensor format
    let mut normalized_tensor = TensorTrait::<
        FP16x16
    >::new(shape: array![tensor_data.data.len()].span(), data: normalized_array.span());
    return normalized_tensor;
}


