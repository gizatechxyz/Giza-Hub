use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};
use orion::numbers::{FixedTrait, FP16x16};

fn get_node_fc3_bias() -> Tensor<FP16x16> {
    let mut shape = array![1];

    let mut data = array![FP16x16 { mag: 228, sign: true }];

    TensorTrait::new(shape.span(), data.span())
}