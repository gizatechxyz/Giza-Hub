// use traits::Into;
use debug::PrintTrait;
use array::{ArrayTrait, SpanTrait};


use linear_regresion::datasets::linear_data::x_feature_data::x_feature_data;
use linear_regresion::datasets::linear_data::y_label_data::y_label_data;


use orion::numbers::{FP16x16,  FixedTrait};
use linear_regresion::model::linear_regression_model::{
     LinearRegressionModel, compute_mean, LinearRegression, LinearRegressionModelTrait
};

use linear_regresion::data_preprocessing::{Dataset, DatasetTrait};
use linear_regresion::helper_functions::{get_tensor_data_by_row, transpose_tensor, calculate_mean , 
calculate_r_score, normalize_user_x_inputs, rescale_predictions};

use orion::operators::tensor::{
    Tensor, TensorTrait, FP16x16Tensor, U32Tensor, U32TensorAdd, 
    FP16x16TensorSub, FP16x16TensorAdd, FP16x16TensorDiv, FP16x16TensorMul};

#[test]
#[available_gas(99999999999999999)]
fn multiple_linear_regression_test() {


// // ----------------------------------------------------------------Simple Linear regression tests---------------------------------------------------------------------------------

let mut main_x_vals = x_feature_data();
let mut main_y_vals = y_label_data();
let dataset = Dataset{x_values: main_x_vals,y_values:main_y_vals};
let mut model = LinearRegression(dataset);
let gradient = model.gradient;
let mut reconstructed_ys = model.predict(main_x_vals);
let mut r_squared_score = calculate_r_score(main_y_vals,reconstructed_ys);
r_squared_score.print(); 

// performing checks on shape on coefficient values (gradient vals + bias) 
assert(model.gradient.data.len() == 1,  'gradient data shape mismatch');
assert(model.bias.data.len() == 1,  'bias data shape mismatch');
// model accuracy deviance checks
assert(r_squared_score >= FixedTrait::new(62259, false), 'Linear model acc. less than 95%');


// linear regression model new input predictions
let mut user_value =   TensorTrait::<FP16x16>::new(shape: array![2].span(), data: array![FixedTrait::new(65536, false), FixedTrait::new(65536, true)].span());
let mut prediction_results = model.predict(user_value);
(*prediction_results.data.at(0)).print(); 
(*prediction_results.data.at(1)).print();


}

