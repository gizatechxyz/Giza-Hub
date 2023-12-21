
use orion::operators::tensor::{
    Tensor, TensorTrait, FP16x16Tensor, U32Tensor, U32TensorAdd, FP16x16TensorSub, FP16x16TensorAdd,
    FP16x16TensorDiv, FP16x16TensorMul
};
use orion::numbers::{FP16x16, FixedTrait};
use multiple_linear_regresion::data_preprocessing::{Dataset, DatasetTrait};
use multiple_linear_regresion::helper_functions::{
    get_tensor_data_by_row, transpose_tensor, calculate_mean, calculate_r_score,
    normalize_user_x_inputs, rescale_predictions
};


#[derive(Copy, Drop)]
struct MultipleLinearRegressionModel {
    coefficients: Tensor<FP16x16>
}

#[generate_trait]
impl RegressionOperation of MultipleLinearRegressionModelTrait {
    // reconstruct the y values using the computed gradients and x values
    fn predict(
        ref self: MultipleLinearRegressionModel, feature_inputs: Tensor<FP16x16>
    ) -> Tensor<FP16x16> {
        // random tensor value that we will replace
        let mut prediction_result = TensorTrait::<
            FP16x16
        >::new(shape: array![1].span(), data: array![FixedTrait::new(10, false)].span());

        let mut result = ArrayTrait::<FP16x16>::new();
        // for multiple predictions
        if feature_inputs.shape.len() > 1 {
            let feature_values = add_bias_term(feature_inputs, 1);
            let mut data_len: u32 = *feature_values.shape.at(0);
            let mut i: u32 = 0;
            loop {
                if i >= data_len {
                    break ();
                }
                let feature_row_values = get_tensor_data_by_row(feature_values, i);
                let predicted_values = feature_row_values.matmul(@self.coefficients);
                result.append(*predicted_values.data.at(0));
                i += 1;
            };
            prediction_result =
                TensorTrait::<
                    FP16x16
                >::new(shape: array![result.len()].span(), data: result.span());
        }

        // for single predictions 
        if feature_inputs.shape.len() == 1 && self.coefficients.data.len() > 1 {
            let feature_values = add_bias_term(feature_inputs, 1);
            prediction_result = feature_values.matmul(@self.coefficients);
        }

        return prediction_result;
    }
}

fn MultipleLinearRegression(dataset: Dataset) -> MultipleLinearRegressionModel {
    let x_values_tranposed = transpose_tensor(dataset.x_values);
    let x_values_tranposed_with_bias = add_bias_term(x_values_tranposed, 0);
    let decorrelated_x_features = decorrelate_x_features(x_values_tranposed_with_bias);
    let coefficients = compute_gradients(
        decorrelated_x_features, dataset.y_values, x_values_tranposed_with_bias
    );
    return MultipleLinearRegressionModel { coefficients };
}

//Adds bias term to features based on axis
fn add_bias_term(x_feature: Tensor<FP16x16>, axis: u32) -> Tensor<FP16x16> {
    let mut x_feature_ = x_feature;
    let mut tensor_with_bias = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::new(10, false)].span());
    let mut result = ArrayTrait::<FP16x16>::new();
    // check if feature data has multiple rows and columns
    if x_feature.shape.len() > 1 {
        let mut index: u32 = 0;
        if axis == 1 {
            index = 0;
        } else {
            index = 1;
        }
        let data_len = *x_feature.shape.at(index); // 50
        let mut i: u32 = 0;
        loop {
            if i >= data_len {
                break ();
            }
            result
                .append(FixedTrait::new(65536, false)); //65536=ONE in FP16x16, change accordingly  
            i += 1;
        };
        if axis == 0 {
            let res_tensor = TensorTrait::new(
                shape: array![1, data_len].span(), data: result.span()
            );
            tensor_with_bias =
                TensorTrait::concat(tensors: array![x_feature, res_tensor].span(), axis: axis);
        } else {
            let res_tensor = TensorTrait::new(
                shape: array![data_len, 1].span(), data: result.span()
            );
            tensor_with_bias =
                TensorTrait::concat(tensors: array![x_feature, res_tensor].span(), axis: axis);
        }
    }
    // check if feature data is 1D
    if x_feature.shape.len() == 1 {
        let mut j: u32 = 0;
        loop {
            match x_feature_.data.pop_front() {
                Option::Some(x_val) => {
                    result.append(*x_val);
                    j += 1;
                },
                Option::None(_) => { break; }
            };
        };
        result.append(FixedTrait::new(65536, false)); //65536=ONE in FP16x16, change accordingly  
        tensor_with_bias =
            TensorTrait::<FP16x16>::new(shape: array![result.len()].span(), data: result.span());
    }
    return tensor_with_bias;
}

// decorrelates the feature data (*only the last tensor row of the decorelated feature data will be fully orthogonal)
fn decorrelate_x_features(x_feature_data: Tensor<FP16x16>) -> Tensor<FP16x16> {
    let mut input_tensor = x_feature_data;

    let mut i: u32 = 0;
    loop {
        if i >= *x_feature_data.shape.at(0) {
            break ();
        }
        let mut placeholder = ArrayTrait::<FP16x16>::new();
        let mut feature_row_values = get_tensor_data_by_row(input_tensor, i);
        let mut feature_squared = feature_row_values.matmul(@feature_row_values);
        // avoiding division by zero errors
        if *feature_squared.data.at(0) == FixedTrait::new(0, false) {
            feature_squared =
                TensorTrait::<
                    FP16x16
                >::new(shape: array![1].span(), data: array![FixedTrait::new(10, false)].span());
        }
        // loop throgh remaining tensor data and remove the individual tensor factors from one another 
        let mut j: u32 = i + 1;
        loop {
            if j >= *x_feature_data.shape.at(0) {
                break ();
            }
            let mut remaining_tensor_values = get_tensor_data_by_row(input_tensor, j);
            let feature_cross_product = feature_row_values.matmul(@remaining_tensor_values);
            let feature_gradients = feature_cross_product / feature_squared;
            remaining_tensor_values = remaining_tensor_values
                - (feature_row_values
                    * feature_gradients); //remove the feature factors from one another
            // loop and append the modifieed remaining_tensor_values (after the corelated factor has been removed) to placeholder array
            let mut k: u32 = 0;
            loop {
                if k >= remaining_tensor_values.data.len() {
                    break ();
                }
                placeholder.append(*remaining_tensor_values.data.at(k));
                k += 1;
            };

            j += 1;
        };
        // convert placeholder array to tensor format and update the original tensor with the new modified decorrelated tensor row values
        let mut decorrelated_tensor = TensorTrait::new(
            shape: array![*x_feature_data.shape.at(0) - 1 - i, *x_feature_data.shape.at(1)].span(),
            data: placeholder.span()
        );
        let mut original_tensor = input_tensor
            .slice(
                starts: array![0, 0].span(),
                ends: array![i + 1, *x_feature_data.shape.at(1)].span(),
                axes: Option::None(()),
                steps: Option::Some(array![1, 1].span())
            );
        input_tensor =
            TensorTrait::concat(
                tensors: array![original_tensor, decorrelated_tensor].span(), axis: 0
            );
        i += 1;
    };
    return input_tensor;
}

// computes the corresponding MLR gradient using decorrelated feature
fn compute_gradients(
    decorrelated_x_features: Tensor<FP16x16>,
    y_values: Tensor<FP16x16>,
    original_x_tensor_values: Tensor<FP16x16>
) -> Tensor<FP16x16> {
    let mut gradient_values_flipped = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::new(10, false)].span());

    let mut result = ArrayTrait::<FP16x16>::new();
    let mut tensor_y_vals = y_values;
    let mut i: u32 = *decorrelated_x_features.shape.at(0);
    // loop through Decorrelated_x_features starting from the fully orthogonlised last tensor row value
    loop {
        if i <= 0 {
            break ();
        }
        let index_val = i - 1;
        let mut decorelated_feature_row_values = get_tensor_data_by_row(
            decorrelated_x_features, index_val
        ); // 50 vals
        let mut decorelated_features_squared = decorelated_feature_row_values
            .matmul(@decorelated_feature_row_values);
        let mut feature_label_cross_product = tensor_y_vals
            .matmul(@decorelated_feature_row_values); // multiply the tensors
        // avoiding division by zero errors
        if *decorelated_features_squared.data.at(0) == FixedTrait::new(0, false) {
            decorelated_features_squared =
                TensorTrait::<
                    FP16x16
                >::new(shape: array![1].span(), data: array![FixedTrait::new(10, false)].span());
        }
        // computing the feature gradient values using the y values and decorrelated x features and appending to array
        let mut single_gradient_value = feature_label_cross_product
            / decorelated_features_squared; // devide the summed value by each other
        result.append(*single_gradient_value.data.at(0));
        // remove the assosciated feature gradient value away from y values
        let mut original_x_tensor_row_values = get_tensor_data_by_row(
            original_x_tensor_values, index_val
        );
        tensor_y_vals = tensor_y_vals
            - (original_x_tensor_row_values
                * single_gradient_value); //remove the first feature from second feature values
        i -= 1;
    };
    // convert the gradient array to tensor format
    let final_gradients = TensorTrait::new(
        shape: array![*decorrelated_x_features.shape.at(0)].span(), data: result.span()
    );

    let mut reverse_grad_array = ArrayTrait::<FP16x16>::new();
    let mut data_len: u32 = final_gradients.data.len();
    loop {
        if data_len <= 0 {
            break ();
        }
        let temp_val = data_len - 1;
        reverse_grad_array.append(*final_gradients.data.at(temp_val));
        data_len -= 1;
    };
    // convert gradient values to tensor format
    let gradient_values_flipped = TensorTrait::<
        FP16x16
    >::new(shape: array![reverse_grad_array.len()].span(), data: reverse_grad_array.span());

    return gradient_values_flipped;
}






 
