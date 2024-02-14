use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};
use orion::numbers::{FixedTrait, FP16x16};

fn get_node_fc3_weight() -> Tensor<FP16x16> {
    let mut shape = array![1, 32];

    let mut data = array![FP16x16 { mag: 53473, sign: true }, FP16x16 { mag: 14036, sign: true }, FP16x16 { mag: 26033, sign: true }, FP16x16 { mag: 28909, sign: true }, FP16x16 { mag: 17369, sign: true }, FP16x16 { mag: 482, sign: true }, FP16x16 { mag: 39352, sign: true }, FP16x16 { mag: 27478, sign: false }, FP16x16 { mag: 36186, sign: true }, FP16x16 { mag: 36328, sign: false }, FP16x16 { mag: 21538, sign: true }, FP16x16 { mag: 14325, sign: true }, FP16x16 { mag: 21849, sign: true }, FP16x16 { mag: 29699, sign: true }, FP16x16 { mag: 18129, sign: true }, FP16x16 { mag: 26106, sign: true }, FP16x16 { mag: 40532, sign: false }, FP16x16 { mag: 46521, sign: false }, FP16x16 { mag: 10749, sign: true }, FP16x16 { mag: 20812, sign: true }, FP16x16 { mag: 24852, sign: true }, FP16x16 { mag: 25180, sign: false }, FP16x16 { mag: 32477, sign: false }, FP16x16 { mag: 3603, sign: false }, FP16x16 { mag: 23305, sign: true }, FP16x16 { mag: 18007, sign: true }, FP16x16 { mag: 38917, sign: false }, FP16x16 { mag: 24724, sign: false }, FP16x16 { mag: 31507, sign: false }, FP16x16 { mag: 20591, sign: true }, FP16x16 { mag: 38031, sign: false }, FP16x16 { mag: 39577, sign: true }];

    TensorTrait::new(shape.span(), data.span())
}