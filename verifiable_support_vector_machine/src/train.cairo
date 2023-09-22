use debug::PrintTrait;
use traits::TryInto;
use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{
    core::{Tensor, TensorTrait, ExtraParams},
    implementations::impl_tensor_fp::{
        Tensor_fp, FixedTypeTensorAdd, FixedTypeTensorMul, FixedTypeTensorSub, FixedTypeTensorDiv
    }
};
use orion::numbers::fixed_point::{
    core::{FixedTrait, FixedType, FixedImpl},
    implementations::fp16x16::core::{
        HALF, ONE, FP16x16Impl, FP16x16Div, FP16x16Print, FP16x16IntoI32
    }
};

use verifiable_support_vector_machine::{helper::{calculate_loss, calculate_gradient}};

// Performs a training step for each iteration during model training
fn train_step(
    x: @Tensor<FixedType>,
    y: @Tensor<FixedType>,
    w: @Tensor<FixedType>,
    learning_rate: FixedType,
    one_tensor: @Tensor<FixedType>,
    half_tensor: @Tensor<FixedType>,
    neg_one_tensor: @Tensor<FixedType>,
    y_train_len: u32,
    iterations: u32,
    index: u32
) -> Tensor<FixedType> {
    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };
    let learning_rate_tensor = TensorTrait::new(
        shape: array![1].span(), data: array![learning_rate].span(), extra: Option::Some(extra),
    );

    let c = TensorTrait::new(
        shape: array![1].span(),
        data: array![FP16x16Impl::ONE()].span(),
        extra: Option::Some(extra),
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
    x: @Tensor<FixedType>,
    y: @Tensor<FixedType>,
    init_w: @Tensor<FixedType>,
    learning_rate: FixedType,
    y_train_len: u32,
    iterations: u32
) -> (Tensor<FixedType>, FixedType, FixedType) {
    let iter_w = init_w;

    'Iterations'.print();
    iterations.print();

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };
    let c = TensorTrait::new(
        shape: array![1].span(),
        data: array![FP16x16Impl::ONE()].span(),
        extra: Option::Some(extra),
    );

    let one_tensor = TensorTrait::new(
        shape: array![1].span(),
        data: array![FP16x16Impl::ONE()].span(),
        extra: Option::Some(extra),
    );

    let half_tensor = TensorTrait::new(
        shape: array![1].span(),
        data: array![FixedTrait::new(HALF, false)].span(),
        extra: Option::Some(extra),
    );

    let neg_one_tensor = TensorTrait::new(
        shape: array![1].span(),
        data: array![FixedTrait::new(ONE, true)].span(),
        extra: Option::Some(extra),
    );

    let initial_loss = FixedTrait::ZERO();
    let final_loss = FixedTrait::ZERO();

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
