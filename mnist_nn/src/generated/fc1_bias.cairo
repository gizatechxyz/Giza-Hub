use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor};
use orion::operators::tensor::implementations::impl_tensor_i32;
use orion::numbers::signed_integer::i32::i32;

fn fc1_bias() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(10);
    let mut data = ArrayTrait::<i32>::new();
    data.append(i32 { mag: 2412, sign: false });
    data.append(i32 { mag: 4648, sign: false });
    data.append(i32 { mag: 237, sign: false });
    data.append(i32 { mag: 1246, sign: true });
    data.append(i32 { mag: 3282, sign: false });
    data.append(i32 { mag: 3806, sign: false });
    data.append(i32 { mag: 2249, sign: false });
    data.append(i32 { mag: 3967, sign: false });
    data.append(i32 { mag: 6493, sign: false });
    data.append(i32 { mag: 3008, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
