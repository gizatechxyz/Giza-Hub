use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};
use orion::numbers::{FixedTrait, FP16x16};

fn get_node_fc2_bias() -> Tensor<FP16x16> {
    let mut shape = array![32];

    let mut data = array![FP16x16 { mag: 3939, sign: false }, FP16x16 { mag: 843, sign: false }, FP16x16 { mag: 293, sign: false }, FP16x16 { mag: 2973, sign: false }, FP16x16 { mag: 350, sign: false }, FP16x16 { mag: 3588, sign: true }, FP16x16 { mag: 1270, sign: true }, FP16x16 { mag: 1128, sign: false }, FP16x16 { mag: 3678, sign: false }, FP16x16 { mag: 2121, sign: false }, FP16x16 { mag: 1661, sign: false }, FP16x16 { mag: 387, sign: true }, FP16x16 { mag: 937, sign: false }, FP16x16 { mag: 802, sign: false }, FP16x16 { mag: 2110, sign: true }, FP16x16 { mag: 2447, sign: true }, FP16x16 { mag: 102, sign: false }, FP16x16 { mag: 3149, sign: false }, FP16x16 { mag: 3171, sign: true }, FP16x16 { mag: 160, sign: false }, FP16x16 { mag: 1625, sign: true }, FP16x16 { mag: 1878, sign: true }, FP16x16 { mag: 917, sign: true }, FP16x16 { mag: 1102, sign: true }, FP16x16 { mag: 395, sign: true }, FP16x16 { mag: 835, sign: true }, FP16x16 { mag: 661, sign: false }, FP16x16 { mag: 5026, sign: false }, FP16x16 { mag: 523, sign: false }, FP16x16 { mag: 145, sign: true }, FP16x16 { mag: 2752, sign: false }, FP16x16 { mag: 31, sign: true }];

    TensorTrait::new(shape.span(), data.span())
}