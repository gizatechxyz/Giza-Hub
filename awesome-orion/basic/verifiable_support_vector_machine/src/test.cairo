use traits::TryInto;
use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{
    Tensor, TensorTrait, FP16x16Tensor, FP16x16TensorAdd, FP16x16TensorMul, FP16x16TensorSub,
    FP16x16TensorDiv
};
use orion::numbers::{FixedTrait, FP16x16, FP16x16Impl};
use orion::numbers::fixed_point::implementations::fp16x16::core::{
    HALF, ONE, FP16x16Mul, FP16x16Div, FP16x16IntoI32, FP16x16PartialOrd,
    FP16x16PartialEq
};

use verifiable_support_vector_machine::{
    generated::{X_train::X_train, Y_train::Y_train, X_test::X_test, Y_test::Y_test}, train::{train}
};

use verifiable_support_vector_machine::{helper::{pred, accuracy}};

#[test]
#[available_gas(99999999999999999)]
fn svm_test() {
    let x_train = X_train();
    let x_test = X_test();
    let y_train = Y_train();
    let y_test = Y_test();

    let feature_size = *x_train.shape[1];

    let mut zero_array = ArrayTrait::new();

    let learning_rate = FixedTrait::new(655, false); // 655 is 0.01

    // 50 %
    let average_compare = FixedTrait::new_unscaled(50, false);

    let mut i = 0_u32;
    loop {
        if i >= feature_size {
            break ();
        }
        zero_array.append(FP16x16Impl::ZERO());
        i += 1;
    };

    let initial_w = TensorTrait::new(
        shape: array![feature_size].span(), data: zero_array.span()
    );

    let y_train_len = y_train.data.len();
    let (final_w, initial_loss, final_loss) = train(
        @x_train, @y_train, @initial_w, learning_rate, y_train_len, 100_u32
    );

    let final_y_pred = pred(@x_test, @final_w);
    let average_pred = accuracy(@final_y_pred, @y_test);

    let train_y_pred = pred(@x_train, @final_w);
    let average_train = accuracy(@train_y_pred, @y_train);

    assert(final_loss < initial_loss, 'No decrease in training loss');
    assert(average_pred > average_compare, 'It is better to flip a coin');
    assert(average_train > average_compare, 'It was not a good training');
}
