use debug::PrintTrait;
use traits::TryInto;
use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{
    Tensor, TensorTrait, FP16x16Tensor, FP16x16TensorAdd, FP16x16TensorMul, FP16x16TensorSub,
    FP16x16TensorDiv
};
use orion::numbers::{FixedTrait, FP16x16, FP16x16Impl};
use orion::numbers::fixed_point::implementations::fp16x16::core::{
    HALF, ONE, FP16x16Mul, FP16x16Div, FP16x16Print, FP16x16IntoI32, FP16x16PartialOrd,
    FP16x16PartialEq
};

use verifiable_support_vector_machine::{helper::{calculate_loss, calculate_gradient}};

// Performs a training step for each iteration during model training
fn train_step(
    x: @Tensor<FP16x16>,
    y: @Tensor<FP16x16>,
    w: @Tensor<FP16x16>,
    learning_rate: FP16x16,
    one_tensor: @Tensor<FP16x16>,
    half_tensor: @Tensor<FP16x16>,
    neg_one_tensor: @Tensor<FP16x16>,
    y_train_len: u32,
    iterations: u32,
    index: u32
) -> Tensor<FP16x16> {
    let learning_rate_tensor = TensorTrait::new(
        shape: array![1].span(), data: array![learning_rate].span()
    );

    let c = TensorTrait::new(
        shape: array![1].span(),
        data: array![FP16x16Impl::ONE()].span(),
    );

    let mut w_recursive = *w;

    let gradient = calculate_gradient(
        @w_recursive, x, y, c, one_tensor, neg_one_tensor, y_train_len
    );

    w_recursive = w_recursive - (learning_rate_tensor * gradient);

    if index == iterations {
        return w_recursive;
    }

    train_step(
        x,
        y,
        @w_recursive,
        learning_rate,
        one_tensor,
        half_tensor,
        neg_one_tensor,
        y_train_len,
        iterations,
        index + 1
    )
}

// Trains the machine learning model.
fn train(
    x: @Tensor<FP16x16>,
    y: @Tensor<FP16x16>,
    init_w: @Tensor<FP16x16>,
    learning_rate: FP16x16,
    y_train_len: u32,
    iterations: u32
) -> (Tensor<FP16x16>, FP16x16, FP16x16) {
    let iter_w = init_w;

    'Iterations'.print();
    iterations.print();

    let c = TensorTrait::new(
        shape: array![1].span(),
        data: array![FP16x16Impl::ONE()].span(),
    );

    let one_tensor = TensorTrait::new(
        shape: array![1].span(),
        data: array![FP16x16Impl::ONE()].span(),
    );

    let half_tensor = TensorTrait::new(
        shape: array![1].span(),
        data: array![FixedTrait::new(HALF, false)].span(),
    );

    let neg_one_tensor = TensorTrait::new(
        shape: array![1].span(),
        data: array![FixedTrait::new(ONE, true)].span(),
    );

    let initial_loss = FixedTrait::<FP16x16>::ZERO();
    let final_loss = FixedTrait::<FP16x16>::ZERO();

    let initial_loss = calculate_loss(init_w, x, y, @c, @one_tensor, @half_tensor, y_train_len);

    let iter_w = train_step(
        x,
        y,
        init_w,
        learning_rate,
        @one_tensor,
        @half_tensor,
        @neg_one_tensor,
        y_train_len,
        iterations,
        1
    );

    let final_loss = calculate_loss(@iter_w, x, y, @c, @one_tensor, @half_tensor, y_train_len);

    (iter_w, initial_loss, final_loss)
}
