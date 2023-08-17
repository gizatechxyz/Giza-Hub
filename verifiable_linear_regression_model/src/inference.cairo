use core::array::SpanTrait;
use traits::Into;
use debug::PrintTrait;
use array::ArrayTrait;
use verifiable_linear_regression::generated::X_values::X_values;
use verifiable_linear_regression::generated::Y_values::Y_values;
use verifiable_linear_regression::helper_functions::{calculate_mean, deviation_from_mean, compute_beta, compute_intercept, predict_y_values, compute_mse, calculate_r_score};


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


fn predict() {
    // Fetching the x and y values
    let y_values = Y_values();
    let x_values = X_values();

    let beta_value = compute_beta(x_values,y_values );
    let intercept_value =  compute_intercept(beta_value, x_values, y_values );


    // input your x value in the variable my_x_value # default 17
    let my_x_value= FixedTrait::new_unscaled(17, false );
    let my_predicted_y = beta_value * my_x_value + intercept_value;

    my_predicted_y.print();

}


