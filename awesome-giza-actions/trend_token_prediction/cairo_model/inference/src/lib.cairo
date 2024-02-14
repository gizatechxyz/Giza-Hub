use orion::operators::tensor::{Tensor, TensorTrait};
use orion::operators::tensor::{U32Tensor, I32Tensor, I8Tensor, FP8x23Tensor, FP16x16Tensor, FP32x32Tensor, BoolTensor};
use orion::numbers::{FP8x23, FP16x16, FP32x32};
use orion::operators::nn::{NNTrait, FP16x16NN};

use node_fc1_weight::get_node_fc1_weight;
use node_fc1_bias::get_node_fc1_bias;
use node_fc2_weight::get_node_fc2_weight;
use node_fc2_bias::get_node_fc2_bias;
use node_fc3_weight::get_node_fc3_weight;
use node_fc3_bias::get_node_fc3_bias;

fn main(node_input: Tensor<FP16x16>) -> Tensor<FP16x16> {
let node_fc1_gemm_output_0 = NNTrait::gemm(node_input, get_node_fc1_weight(), Option::Some(get_node_fc1_bias()), Option::Some(FP16x16 { mag: 65536, sign: false }), Option::Some(FP16x16 { mag: 65536, sign: false }), false, true);
let node_relu_output_0 = NNTrait::relu(@node_fc1_gemm_output_0);
let node_fc2_gemm_output_0 = NNTrait::gemm(node_relu_output_0, get_node_fc2_weight(), Option::Some(get_node_fc2_bias()), Option::Some(FP16x16 { mag: 65536, sign: false }), Option::Some(FP16x16 { mag: 65536, sign: false }), false, true);
let node_relu_1_output_0 = NNTrait::relu(@node_fc2_gemm_output_0);
let node_fc3_gemm_output_0 = NNTrait::gemm(node_relu_1_output_0, get_node_fc3_weight(), Option::Some(get_node_fc3_bias()), Option::Some(FP16x16 { mag: 65536, sign: false }), Option::Some(FP16x16 { mag: 65536, sign: false }), false, true);
let node_output = FP16x16NN::sigmoid(@node_fc3_gemm_output_0);

        node_output
    }