use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor};
use orion::operators::tensor::implementations::impl_tensor_i32;
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};

fn fc2_bias() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(10);
    let mut data = ArrayTrait::<i32>::new();
    data.append(IntegerTrait::new(48, true));
    data.append(IntegerTrait::new(333, true));
    data.append(IntegerTrait::new(212, true));
    data.append(IntegerTrait::new(217, true));
    data.append(IntegerTrait::new(70, false));
    data.append(IntegerTrait::new(89, true));
    data.append(IntegerTrait::new(260, true));
    data.append(IntegerTrait::new(348, true));
    data.append(IntegerTrait::new(735, false));
    data.append(IntegerTrait::new(58, false));
    TensorTrait::new(shape.span(), data.span())
}
