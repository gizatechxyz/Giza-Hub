use orion::operators::tensor::core::TensorTrait;

use core::array::{SpanTrait, ArrayTrait};

use mnist_nn::nn::fc1;
use mnist_nn::nn::fc2;
use mnist_nn::generated::input::input;
use mnist_nn::generated::fc1_bias::fc1_bias;
use mnist_nn::generated::fc1_weights::fc1_weights;
use mnist_nn::generated::fc2_bias::fc2_bias;
use mnist_nn::generated::fc2_weights::fc2_weights;

use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;

fn main() -> u32 {
    let input = input();
    let fc1_bias = fc1_bias();
    let fc1_weights = fc1_weights();
    let fc2_bias = fc2_bias();
    let fc2_weights = fc2_weights();

    let x = fc1(input, fc1_weights, fc1_bias);
    let x = fc2(x, fc2_weights, fc2_bias);
    
    let x = *x.argmax(0, Option::None(()), Option::None(())).data.at(0);

    x
}