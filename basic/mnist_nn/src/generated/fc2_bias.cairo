use array::ArrayTrait;
use orion::operators::tensor::{TensorTrait, Tensor, I32Tensor};
use orion::numbers::i32;


fn fc2_bias() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(10);
    let mut data = ArrayTrait::<i32>::new();
    data.append(i32 { mag: 313, sign: true });
    data.append(i32 { mag: 1064, sign: false });
    data.append(i32 { mag: 28, sign: true });
    data.append(i32 { mag: 184, sign: true });
    data.append(i32 { mag: 1012, sign: true });
    data.append(i32 { mag: 1885, sign: false });
    data.append(i32 { mag: 787, sign: true });
    data.append(i32 { mag: 835, sign: false });
    data.append(i32 { mag: 1819, sign: true });
    data.append(i32 { mag: 208, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
