use array::ArrayTrait;
use orion::operators::tensor::{TensorTrait, Tensor, I32Tensor};
use orion::numbers::i32;

fn fc2_bias() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(10);
    let mut data = ArrayTrait::<i32>::new();
    data.append(i32 { mag: 366, sign: true });
    data.append(i32 { mag: 1347, sign: false });
    data.append(i32 { mag: 353, sign: false });
    data.append(i32 { mag: 1007, sign: true });
    data.append(i32 { mag: 14, sign: false });
    data.append(i32 { mag: 1041, sign: false });
    data.append(i32 { mag: 341, sign: true });
    data.append(i32 { mag: 901, sign: false });
    data.append(i32 { mag: 1768, sign: true });
    data.append(i32 { mag: 76, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
