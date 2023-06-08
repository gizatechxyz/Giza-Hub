use core::array::SpanTrait;
use core::debug::PrintTrait;
use mnist_nn::nn::fc1;
use mnist_nn::nn::fc2;

use mnist_nn::generated::input::input;
use mnist_nn::generated::fc1_bias::fc1_bias;
use mnist_nn::generated::fc1_weights::fc1_weights;
use mnist_nn::generated::fc2_bias::fc2_bias;
use mnist_nn::generated::fc2_weights::fc2_weights;


#[test]
#[available_gas(99999999999999999)]
fn mnist_nn_test() {
    let input = input();
    let fc1_bias = fc1_bias();
    let fc1_weights = fc1_weights();
    let fc2_bias = fc2_bias();
    let fc2_weights = fc2_weights();

    let x = fc1(input, fc1_weights, fc1_bias);
    let x = fc2(x, fc2_weights, fc2_bias);

    assert(*x.data.at(0).mag == 0, 'proba x is 0 -> 0');
    assert(*x.data.at(1).mag == 0, 'proba x is 1 -> 0');
    assert(*x.data.at(2).mag == 0, 'proba x is 2 -> 0');
    assert(*x.data.at(3).mag == 0, 'proba x is 3 -> 0');
    assert(*x.data.at(4).mag == 0, 'proba x is 4 -> 0');
    assert(*x.data.at(5).mag == 0, 'proba x is 5 -> 0');
    assert(*x.data.at(6).mag == 0, 'proba x is 6 -> 0');
    assert(*x.data.at(7).mag > 62259, 'proba x is 7 -> 1');
    assert(*x.data.at(8).mag == 0, 'proba x is 8 -> 0');
    assert(*x.data.at(9).mag == 0, 'proba x is 9 -> 0');
}
