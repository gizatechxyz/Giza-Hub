use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor};
use orion::operators::tensor::implementations::impl_tensor_i32;
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};

fn fc2_bias() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(10);
    let mut data = ArrayTrait::<i32>::new();
    data.append(IntegerTrait::new(397, true));
    data.append(IntegerTrait::new(1384, false));
    data.append(IntegerTrait::new(325, false));
    data.append(IntegerTrait::new(942, true));
    data.append(IntegerTrait::new(434, true));
    data.append(IntegerTrait::new(956, false));
    data.append(IntegerTrait::new(174, true));
    data.append(IntegerTrait::new(1310, false));
    data.append(IntegerTrait::new(1651, true));
    data.append(IntegerTrait::new(2, false));
    TensorTrait::new(shape.span(), data.span())
}
