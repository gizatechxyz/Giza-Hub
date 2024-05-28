use orion::operators::tensor::{Tensor, TensorTrait};
use orion::operators::nn::{NNTrait, FP16x16NN};
use orion::operators::tensor::{U32Tensor, I32Tensor, I8Tensor, FP8x23Tensor, FP16x16Tensor, FP32x32Tensor, BoolTensor};
use orion::numbers::{FP8x23, FP16x16, FP32x32, FixedTrait};
use orion::operators::matrix::{MutMatrix, MutMatrixImpl};

fn main(node_input: Tensor<FP16x16>) -> Tensor<FP16x16> {
    let kappa = FixedTrait::from_felt(10);
    return NNTrait::threshold_relu(@node_input, @kappa);
}