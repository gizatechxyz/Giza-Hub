use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor};
use orion::operators::tensor::implementations::impl_tensor_i32;
use orion::numbers::signed_integer::i32::i32;

fn fc2_bias() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(10);
    let mut data = ArrayTrait::<i32>::new();
    data.append(i32 { mag: 891, sign: true });
    data.append(i32 { mag: 729, sign: false });
    data.append(i32 { mag: 252, sign: false });
    data.append(i32 { mag: 869, sign: true });
    data.append(i32 { mag: 34, sign: true });
    data.append(i32 { mag: 741, sign: false });
    data.append(i32 { mag: 63, sign: false });
    data.append(i32 { mag: 518, sign: false });
    data.append(i32 { mag: 811, sign: true });
    data.append(i32 { mag: 90, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
