use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn tensor() -> Tensor<FP16x16> {

   Tensor {
       shape: array![9,].span(),
       data: array![
FP16x16 {mag: 1241, sign: true}, FP16x16 {mag: 32808, sign: true}, FP16x16 {mag: 5666, sign: false}, FP16x16 {mag: 896, sign: false}, FP16x16 {mag: 12018, sign: false}, FP16x16 {mag: 11826, sign: false}, FP16x16 {mag: 3795, sign: false}, FP16x16 {mag: 13949, sign: false}, FP16x16 {mag: 3218, sign: true}, ].span()
   }
}
