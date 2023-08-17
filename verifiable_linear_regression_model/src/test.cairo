use core::array::SpanTrait;
use traits::Into;
use debug::PrintTrait;
use array::ArrayTrait;
use verifiable_linear_regression::generated::X_values::X_values;
use verifiable_linear_regression::generated::Y_values::Y_values;
use verifiable_linear_regression::lin_reg_func::{calculate_mean, deviation_from_mean, compute_beta, compute_intercept, predict_y_values, compute_mse, calculate_r_score};

use orion::operators::tensor::math::cumsum::cumsum_i32::cumsum;
use orion::operators::tensor::implementations::{impl_tensor_u32::Tensor_u32, impl_tensor_fp::Tensor_fp};
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::operators::tensor::math::arithmetic::arithmetic_fp::core::{add, sub, mul, div};
use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};
use orion::numbers::fixed_point::implementations::impl_16x16::{
    FP16x16Impl, FP16x16Add, FP16x16AddEq, FP16x16Into, FP16x16Print, FP16x16PartialEq, FP16x16Sub,
    FP16x16SubEq, FP16x16Mul, FP16x16MulEq, FP16x16Div, FP16x16DivEq, FP16x16PartialOrd, FP16x16Neg
};

use orion::operators::tensor::linalg::matmul::matmul_fp::core::matmul;


#[test]
#[available_gas(99999999999999999)]
fn linear_regression_test() {
    // Fetching the x and y values
    let y_values = Y_values();
    let x_values = X_values();

    // (*x_values.data.at(18)).print();

    let beta_value = compute_beta(x_values,y_values );
    // beta_value.print();    // calculated gradient value

    let intercept_value =  compute_intercept(beta_value, x_values, y_values );
    // intercept_value.print();   // calculated intercept value

    let y_pred = predict_y_values(beta_value, x_values, y_values );

    let mse = compute_mse(y_values, y_pred);
    // mse.print();       // mean squared error ouput

    let r_score = calculate_r_score(y_values, y_pred);
    r_score.print();   // accuracy of model around 0.8303375244140625

    assert(beta_value.mag > 0, 'x & y not positively correlated');
    assert(r_score.mag > 0, 'R-Squared needs to be above 0');
    assert(r_score.mag < 62259, 'R-Squared has to be below 65536'); // 65536 represents ONE in fp16x16.
    assert(r_score.mag > 32768, 'Accuracy below 50% ');
}

