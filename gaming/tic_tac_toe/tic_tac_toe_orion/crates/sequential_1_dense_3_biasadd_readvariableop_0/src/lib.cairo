use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn tensor() -> Tensor<FP16x16> {

   Tensor {
       shape: array![1,].span(),
       data: array![
FP16x16 {mag: 17046, sign: false}, ].span()
   }
}
