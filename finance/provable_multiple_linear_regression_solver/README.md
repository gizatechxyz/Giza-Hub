# A Closed-Form Provable Multiple Linear Regression Solver in Cairo

## Provability and Verifiability:
The key benefit of this Lightweight Multiple Linear Regression Solver lies in its commitment to <b>Provability and Verifiability</b>. By utilizing <b>[Cairo](https://www.cairo-lang.org/) & [Orion](https://github.com/gizatechxyz/orion)</b>, the entire MLR system becomes inherently provable through [STARKs](https://starkware.co/stark/), ensuring unparalleled transparency and trustworthiness. This enables for every inference of the model construction, execution and prediction phase to be transparently proved using e.g LambdaClass STARK Prover. In essence, the Provability and Verifiability aspect ensures that the tool is not only for prediction but also a framework to build accountability and trust in on-chain business environments. 

## Overview:
In many data-oriented business applications, Multiple Linear Regression remains a powerful tool for problem-solving. As we step into the <b>ProvableML</b> domain to enhance model transparency, these algorithms still prove to be advantageous in on-chain environments due to their lightweight, interpretable, and cost-efficient attributes. 

Traditionally, the common approach to Multiple Linear Regression (MLR) involves computing pseudo-inverses and Singular Value Decomposition (<b>SVD</b>). While robust, their implementation complexity can often overshadow the regression problem at hand. Consequently, <b>gradient-based methods</b> are often preferred in data science projects, but this also can be deemed excessive due to the resource-intensive iterative approach to approximate gradients and the manual hyperparameter tuning required. This can be a hindrance, especially in automated on-chain environments and can also be fairly costly too.

## Closed-Form Multiple Linear Regression Solver for StarkNet
In light of these considerations, this repository introduces an <b>intuitive closed-form approach  to calculating MLR gradients without any hyperparameter tuning</b>, making it easy to estimate computational steps/cost required given a dataset, unlike gradient-based methods.

The MLR comprises of three integral components:\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>1.Orthogonalization of Input Features:</b> Ensures independence among the X features.\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>2.Gradient Calculation:</b> Computes the exact gradient  between each decorrelated X feature and y variable.\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>3.Forecasting & Predictions:</b> Utilizes the computed coefficients to make new predictions.

This relatively intuitive approach makes it a simple model to interpret, integrate, as well as debug, thereby reducing the trust barrier for both builders and end-users interacting with MLR systems in Starknet. 

## How to use it
```rust
// import necessary orion libs and your x and y values 
use debug::PrintTrait;
use array::{ArrayTrait, SpanTrait};
use multiple_linear_regresion::datasets::aave_data::aave_x_features::aave_x_features;
use multiple_linear_regresion::datasets::aave_data::aave_y_labels::aave_y_labels; 
use multiple_linear_regresion::model::multiple_linear_regression_model::{MultipleLinearRegressionModel, MultipleLinearRegression, MultipleLinearRegressionModelTrait};
use multiple_linear_regresion::data_preprocessing::{Dataset, DatasetTrait};
use multiple_linear_regresion::helper_functions::{get_tensor_data_by_row, transpose_tensor, calculate_mean , calculate_r_score, normalize_user_x_inputs, rescale_predictions};
use orion::numbers::{FP16x16,  FixedTrait};
use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor, U32Tensor, U32TensorAdd, FP16x16TensorSub, FP16x16TensorAdd, FP16x16TensorDiv, FP16x16TensorMul};


// Dataset instance is instantiated by passing the x and y values
let mut main_x_vals = aave_x_features();
let mut main_y_vals = aave_y_labels();
let mut dataset = Dataset{x_values: main_x_vals,y_values:main_y_vals};

// dataset is normalized using built-in method to avoid overflow issues in subsequent steps
let mut normalized_dataset = dataset.normalize_dataset();

// instantiate MultipleLinearRegression and pass the normalized dataset. This will fit the model to the provided dataset.
let mut model = MultipleLinearRegression(normalized_dataset);
// access the model coefficients using the following built-in method
let mut model_coefficients = model.coefficients; 
// make new predictions using the constructed model 
let mut predictions = model.predict (new_x_values);

// Computing the training accuracy to assess model perfomance
let mut reconstructed_ys = model.predict (normalized_dataset.x_values);
let mut r_squared_score = calculate_r_score(normalized_dataset.y_values,reconstructed_ys);
r_squared_score.print(); 
```
### Tutorial walkthroughs
To provide a deeper understanding of how the MLR solver works several notebook tutorials have been implemented as a walkthrough example 😁. Some of the examples include:
- [Forecasting AAVE's WETH Pool Lifetime Repayments](https://github.com/BemTG/Provable-Multiple-Linear-Regression-Solver/blob/main/notebook%20tutorials/Provable%20Multiple%20Linear%20Regression%20Solver%20(Forecasting%20AAVE%20Business%20metrics).ipynb)
- Boston Housing Dataset: Estimates valuation of house prices given multiple feature inputs
- Basic Simple Linear regression dataset
