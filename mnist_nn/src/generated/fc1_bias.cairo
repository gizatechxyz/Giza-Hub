use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor};
use orion::operators::tensor::implementations::impl_tensor_i32;
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};

fn fc1_bias() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(10);
    let mut data = ArrayTrait::<i32>::new();
    data.append(IntegerTrait::new(7981, false));
    data.append(IntegerTrait::new(6046, false));
    data.append(IntegerTrait::new(2233, false));
    data.append(IntegerTrait::new(6356, true));
    data.append(IntegerTrait::new(2468, false));
    data.append(IntegerTrait::new(1100, true));
    data.append(IntegerTrait::new(2097, true));
    data.append(IntegerTrait::new(3441, false));
    data.append(IntegerTrait::new(5365, false));
    data.append(IntegerTrait::new(1672, false));
    TensorTrait::new(shape.span(), data.span())
}
