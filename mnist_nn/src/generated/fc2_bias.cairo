use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor};
use orion::operators::tensor::implementations::impl_tensor_i32;
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};

fn fc2_bias() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(10);
    let mut data = ArrayTrait::<i32>::new();
    data.append(IntegerTrait::new(111, true));
    data.append(IntegerTrait::new(757, false));
    data.append(IntegerTrait::new(339, true));
    data.append(IntegerTrait::new(46, false));
    data.append(IntegerTrait::new(160, false));
    data.append(IntegerTrait::new(341, false));
    data.append(IntegerTrait::new(188, true));
    data.append(IntegerTrait::new(37, false));
    data.append(IntegerTrait::new(1378, true));
    data.append(IntegerTrait::new(467, false));
    TensorTrait::new(shape.span(), data.span())
}
