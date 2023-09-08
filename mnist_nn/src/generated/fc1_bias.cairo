use array::ArrayTrait;
use orion::operators::tensor::{TensorTrait, Tensor, I32Tensor};
use orion::numbers::i32;


fn fc1_bias() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(10);
    let mut data = ArrayTrait::<i32>::new();
    data.append(i32 { mag: 1287, sign: false });
    data.append(i32 { mag: 3667, sign: true });
    data.append(i32 { mag: 2954, sign: false });
    data.append(i32 { mag: 7938, sign: false });
    data.append(i32 { mag: 3959, sign: false });
    data.append(i32 { mag: 5862, sign: true });
    data.append(i32 { mag: 4886, sign: false });
    data.append(i32 { mag: 4992, sign: false });
    data.append(i32 { mag: 10126, sign: false });
    data.append(i32 { mag: 2237, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
