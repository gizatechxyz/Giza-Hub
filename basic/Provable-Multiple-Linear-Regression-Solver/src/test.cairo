// use traits::Into;
use debug::PrintTrait;
use array::{ArrayTrait, SpanTrait};

use multiple_linear_regresion::datasets::aave_data::aave_x_features::aave_x_features;
use multiple_linear_regresion::datasets::aave_data::aave_y_labels::aave_y_labels;
use multiple_linear_regresion::datasets::boston_data::boston_x_features::boston_x_features;
use multiple_linear_regresion::datasets::boston_data::boston_y_labels::boston_y_labels;
use multiple_linear_regresion::datasets::linear_data::feature_data::x_feature_data;
use multiple_linear_regresion::datasets::linear_data::label_data::y_label_data;
use multiple_linear_regresion::datasets::user_inputs_data::user_inputs_boston_data::user_input_boston_housing;
use multiple_linear_regresion::datasets::user_inputs_data::aave_weth_revenue_data_input::{
    aave_weth_revenue_data_input
};

use orion::numbers::{FP16x16, FixedTrait};
use multiple_linear_regresion::model::linear_regression_model::{
    LinearRegressionModel, compute_mean, LinearRegression, LinearRegressionModelTrait
};
use multiple_linear_regresion::model::multiple_linear_regression_model::{
    MultipleLinearRegressionModel, MultipleLinearRegression, MultipleLinearRegressionModelTrait
};
use multiple_linear_regresion::data_preprocessing::{
    Dataset, DatasetTrait, normalize_feature_data, normalize_label_data
};
use multiple_linear_regresion::helper_functions::{
    get_tensor_data_by_row, transpose_tensor, calculate_mean, calculate_r_score,
    normalize_user_x_inputs, rescale_predictions
};

use orion::operators::tensor::{
    Tensor, TensorTrait, FP16x16Tensor, U32Tensor, U32TensorAdd, FP16x16TensorSub, FP16x16TensorAdd,
    FP16x16TensorDiv, FP16x16TensorMul
};

#[test]
#[available_gas(99999999999999999)]
fn multiple_linear_regression_test() {
    // -------------------------------------------------------------------AAVE dataset tests---------------------------------------------------------------------------------------------

    let mut main_x_vals = aave_x_features();
    let mut main_y_vals = aave_y_labels();
    let mut dataset = Dataset { x_values: main_x_vals, y_values: main_y_vals };
    let mut normalized_dataset = dataset.normalize_dataset();
    let mut model = MultipleLinearRegression(normalized_dataset);
    let mut model_coefficients = model.coefficients;
    let mut reconstructed_ys = model.predict(normalized_dataset.x_values);
    let mut r_squared_score = calculate_r_score(normalized_dataset.y_values, reconstructed_ys);
    r_squared_score.print();

    // checking if data has been normalized correctly
    assert(
        normalized_dataset.x_values.max_in_tensor() <= FixedTrait::new(65536, false),
        'normalized x not between 0-1'
    );
    assert(
        normalized_dataset.x_values.min_in_tensor() >= FixedTrait::new(0, false),
        'normalized x not between 0-1'
    );
    assert(
        normalized_dataset.y_values.max_in_tensor() <= FixedTrait::new(65536, false),
        'normalized y not between 0-1'
    );
    assert(
        normalized_dataset.x_values.min_in_tensor() >= FixedTrait::new(0, false),
        'normalized y not between 0-1'
    );
    // performing checks on the shape of normalized data
    assert(
        normalized_dataset.x_values.data.len() == main_x_vals.data.len()
            && normalized_dataset.y_values.data.len() == main_y_vals.data.len(),
        'normalized data shape mismatch'
    );
    // performing checks on shape on coefficient values (gradient vals + bias)
    assert(
        model.coefficients.data.len() == *main_x_vals.shape.at(1) + 1,
        'coefficient data shape mismatch'
    );
    // model accuracy deviance checks
    assert(r_squared_score >= FixedTrait::new(62259, false), 'AAVE model acc. less than 95%');

    // using model to forecast aave's 7 day WETH lifetime repayments forecast 
    let last_7_days_aave_data = aave_weth_revenue_data_input();
    let last_7_days_aave_data_normalized = normalize_user_x_inputs(
        last_7_days_aave_data, main_x_vals
    );
    let mut forecast_results = model.predict(last_7_days_aave_data_normalized);
    let mut rescale_forecasts = rescale_predictions(
        forecast_results, main_y_vals
    ); // PS. ** the rescaled forecasted ouputs are in denominated thousands of ETH
    (*rescale_forecasts.data.at(0)).print();
    (*rescale_forecasts.data.at(1)).print();
    (*rescale_forecasts.data.at(2)).print();
    (*rescale_forecasts.data.at(5)).print();
    (*rescale_forecasts.data.at(6)).print();
// -------------------------------------------------------------------Boston dataset tests---------------------------------------------------------------------------------------------

// let mut main_x_vals = boston_x_features();
// let mut main_y_vals = boston_y_labels();
// let mut dataset = Dataset{x_values: main_x_vals,y_values:main_y_vals};
// let mut normalized_dataset = dataset.normalize_dataset();
// let mut model  = MultipleLinearRegression(normalized_dataset);
// let mut model_coefficients = model.coefficients;
// let mut reconstructed_ys = model.predict (normalized_dataset.x_values);
// let mut r_squared_score = calculate_r_score(normalized_dataset.y_values,reconstructed_ys);
// r_squared_score.print(); 

// // checking if data has been normalized correctly
// assert(normalized_dataset.x_values.max_in_tensor() <= FixedTrait::new(65536, false), 'normalized x not between 0-1');
// assert(normalized_dataset.x_values.min_in_tensor() >= FixedTrait::new(0, false), 'normalized x not between 0-1');
// assert(normalized_dataset.y_values.max_in_tensor() <= FixedTrait::new(65536, false), 'normalized y not between 0-1');
// assert(normalized_dataset.x_values.min_in_tensor() >= FixedTrait::new(0, false), 'normalized y not between 0-1');
// // performing checks on the shape of normalized data
// assert(normalized_dataset.x_values.data.len()== main_x_vals.data.len() && 
// normalized_dataset.y_values.data.len()== main_y_vals.data.len() , 'normalized data shape mismatch');
// // performing checks on shape on coefficient values (gradient vals + bias)
// assert(model.coefficients.data.len() == *main_x_vals.shape.at(1)+1, 'coefficient data shape mismatch');
// // model accuracy deviance checks
// assert(r_squared_score >= FixedTrait::new(55699, false), 'Boston model acc. less than 84%');

// // boston user inputed house valuation predictions
// let user_input = user_input_boston_housing();
// let mut normalized_user_x_inputs = normalize_user_x_inputs(user_input, main_x_vals) ;
// let mut prediction_result  = model.predict (normalized_user_x_inputs); 
// let mut rescale_prediction  = rescale_predictions(prediction_result, main_y_vals);
// (*rescale_prediction.data.at(0)).print(); 

// // ----------------------------------------------------------------Simple Linear regression tests---------------------------------------------------------------------------------

// let mut main_x_vals = x_feature_data();
// let mut main_y_vals = y_label_data();
// let dataset = Dataset{x_values: main_x_vals,y_values:main_y_vals};
// let mut model = LinearRegression(dataset);
// let gradient = model.gradient;
// let mut reconstructed_ys = model.predict(main_x_vals);
// let mut r_squared_score = calculate_r_score(main_y_vals,reconstructed_ys);
// r_squared_score.print(); 

// // // performing checks on shape on coefficient values (gradient vals + bias) 
// // assert(model.gradient.data.len() == 1,  'gradient data shape mismatch');
// // assert(model.bias.data.len() == 1,  'bias data shape mismatch');
// // // model accuracy deviance checks
// // assert(r_squared_score >= FixedTrait::new(62259, false), 'Linear model acc. less than 95%');

// // linear regression model new input predictions
// let mut user_value =   TensorTrait::<FP16x16>::new(shape: array![2].span(), data: array![FixedTrait::new(65536, false), FixedTrait::new(65536, true)].span());
// let mut prediction_results = model.predict(user_value);
// (*prediction_results.data.at(0)).print(); 
// (*prediction_results.data.at(1)).print();

}
