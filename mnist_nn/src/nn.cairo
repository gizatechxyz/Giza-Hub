use orion::operators::tensor::core::Tensor;
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use orion::operators::nn::core::NNTrait;
use orion::numbers::fixed_point::core::FixedType;
use orion::operators::nn::implementations::impl_nn_i32;

fn fc1(i: Tensor<i32>, w: Tensor<i32>, b: Tensor<i32>) -> Tensor<i32> {
    let x = NNTrait::linear(i, w, b, true);
    NNTrait::relu(@x, IntegerTrait::new(0, false))
}

fn fc2(i: Tensor<i32>, w: Tensor<i32>, b: Tensor<i32>) -> Tensor<FixedType> {
    let x = NNTrait::linear(i, w, b, true);
    let x = NNTrait::relu(@x, IntegerTrait::new(0, false));
    NNTrait::softmax(@x, 0)
}
