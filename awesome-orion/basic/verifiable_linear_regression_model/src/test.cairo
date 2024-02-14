use debug::PrintTrait;

use verifiable_linear_regression::generated::X_values::X_values;
use verifiable_linear_regression::generated::Y_values::Y_values;
use verifiable_linear_regression::lin_reg_func::{
    calculate_mean, deviation_from_mean, compute_beta, compute_intercept, predict_y_values,
    compute_mse, calculate_r_score
};


#[test]
#[available_gas(99999999999999999)]
fn linear_regression_test() {
    // Fetching the x and y values
    let y_values = Y_values();
    let x_values = X_values();

    // (*x_values.data.at(18)).print();

    let beta_value = compute_beta(x_values, y_values);
    // beta_value.print();    // calculated gradient value

    let intercept_value = compute_intercept(beta_value, x_values, y_values);
    // intercept_value.print();   // calculated intercept value

    let y_pred = predict_y_values(beta_value, x_values, y_values);

    let mse = compute_mse(y_values, y_pred);
    // mse.print();       // mean squared error ouput

    let r_score = calculate_r_score(y_values, y_pred);
    r_score.print(); // accuracy of model around 0.97494506835

    assert(beta_value.mag > 0, 'x & y not positively correlated');
    assert(r_score.mag > 0, 'R-Squared needs to be above 0');
    assert(
        r_score.mag < 65536, 'R-Squared has to be below 65536'
    ); // 65536 represents ONE in fp16x16.
    assert(r_score.mag > 32768, 'Accuracy below 50% ');
}

