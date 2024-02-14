
// use traits::Into;
use debug::PrintTrait;
use array::{ArrayTrait, SpanTrait};


use multiple_linear_regresion::datasets::boston_data::boston_x_features::boston_x_features;
use multiple_linear_regresion::datasets::boston_data::boston_y_labels::boston_y_labels;
use multiple_linear_regresion::datasets::user_inputs_data::user_inputs_boston_data::user_inputs_boston_data;

use orion::numbers::{FP16x16,  FixedTrait};

use multiple_linear_regresion::model::multiple_linear_regression_model::{
     MultipleLinearRegressionModel, MultipleLinearRegression, MultipleLinearRegressionModelTrait
};
use multiple_linear_regresion::data_preprocessing::{Dataset, DatasetTrait};
use multiple_linear_regresion::helper_functions::{get_tensor_data_by_row, transpose_tensor, calculate_mean , 
calculate_r_score, normalize_user_x_inputs, rescale_predictions};

use orion::operators::tensor::{
    Tensor, TensorTrait, FP16x16Tensor, U32Tensor, U32TensorAdd, 
    FP16x16TensorSub, FP16x16TensorAdd, FP16x16TensorDiv, FP16x16TensorMul};

#[test]
#[available_gas(99999999999999999)]
fn multiple_linear_regression_test() {

// -------------------------------------------------------------------Boston dataset tests---------------------------------------------------------------------------------------------

let mut main_x_vals = boston_x_features();
let mut main_y_vals = boston_y_labels();
let mut dataset = Dataset{x_values: main_x_vals,y_values:main_y_vals};
let mut normalized_dataset = dataset.normalize_dataset();
let mut model  = MultipleLinearRegression(normalized_dataset);
let mut model_coefficients = model.coefficients;
let mut reconstructed_ys = model.predict (normalized_dataset.x_values);
let mut r_squared_score = calculate_r_score(normalized_dataset.y_values,reconstructed_ys);
r_squared_score.print(); 

// checking if data has been normalized correctly
assert(normalized_dataset.x_values.max_in_tensor() <= FixedTrait::new(65536, false), 'normalized x not between 0-1');
assert(normalized_dataset.x_values.min_in_tensor() >= FixedTrait::new(0, false), 'normalized x not between 0-1');
assert(normalized_dataset.y_values.max_in_tensor() <= FixedTrait::new(65536, false), 'normalized y not between 0-1');
assert(normalized_dataset.x_values.min_in_tensor() >= FixedTrait::new(0, false), 'normalized y not between 0-1');
// performing checks on the shape of normalized data
assert(normalized_dataset.x_values.data.len()== main_x_vals.data.len() && 
normalized_dataset.y_values.data.len()== main_y_vals.data.len() , 'normalized data shape mismatch');
// performing checks on shape on coefficient values (gradient vals + bias)
assert(model.coefficients.data.len() == *main_x_vals.shape.at(1)+1, 'coefficient data shape mismatch');
// model accuracy deviance checks
assert(r_squared_score >= FixedTrait::new(55699, false), 'Boston model acc. less than 84%');


// boston user inputed house valuation predictions
let user_input = user_inputs_boston_data();
let mut normalized_user_x_inputs = normalize_user_x_inputs(user_input, main_x_vals) ;
let mut prediction_result  = model.predict (normalized_user_x_inputs); 
let mut rescale_prediction  = rescale_predictions(prediction_result, main_y_vals);
(*rescale_prediction.data.at(0)).print(); 


}

