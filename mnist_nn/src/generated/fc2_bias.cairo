use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor};
use orion::operators::tensor::implementations::impl_tensor_i32;
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};

fn fc2_bias() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(10);
    let mut data = ArrayTrait::<i32>::new();
    data.append(IntegerTrait::new(223, true));
    data.append(IntegerTrait::new(209, true));
    data.append(IntegerTrait::new(352, true));
    data.append(IntegerTrait::new(111, true));
    data.append(IntegerTrait::new(155, false));
    data.append(IntegerTrait::new(114, true));
    data.append(IntegerTrait::new(124, true));
    data.append(IntegerTrait::new(309, true));
    data.append(IntegerTrait::new(576, false));
    data.append(IntegerTrait::new(159, false));
    TensorTrait::new(shape.span(), data.span())
}
