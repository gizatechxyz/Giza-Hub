use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor};
use orion::operators::tensor::implementations::impl_tensor_i32;
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};

fn fc1_bias() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(10);
    let mut data = ArrayTrait::<i32>::new();
    data.append(IntegerTrait::new(8084, false));
    data.append(IntegerTrait::new(80, false));
    data.append(IntegerTrait::new(6392, false));
    data.append(IntegerTrait::new(429, false));
    data.append(IntegerTrait::new(1046, false));
    data.append(IntegerTrait::new(1025, false));
    data.append(IntegerTrait::new(442, false));
    data.append(IntegerTrait::new(7991, false));
    data.append(IntegerTrait::new(4957, true));
    data.append(IntegerTrait::new(4932, false));
    TensorTrait::new(shape.span(), data.span())
}
